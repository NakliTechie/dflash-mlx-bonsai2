// Microbench: engine decode matvec / engine 8-row prefill matmul / spike 8-row WGSL GEMM on layer-0 up_proj (and lm_head),
// all on the engine's own GPU-resident lut2_128 weights, with a float64 CPU reference from the dequantized weights.
// Results land in window.__gs.results; the CDP driver prints them.
window.__gs = { state: 'init', log: [] };
const V = window.__gs; const mark = (m) => { V.log.push(m); V.state = m; };
const q = new URLSearchParams(location.search);
const ITERS = Number(q.get('iters') || 21);
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const fmt = (x) => Number(x.toFixed(4));
function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
// deterministic gaussian rows
function randn(n, seed) {
  let s = seed >>> 0; const out = new Float32Array(n);
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s + 0.5) / 4294967296; };
  for (let i = 0; i < n; ++i) { const u = rnd(), v = rnd(); out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  return out;
}
(async () => {
  try {
    const { gemmWgsl, gemmWgslV2, gemmWgslV3, gemmWgslV4, gemmWgslV5, gemmWgslV6, reduceWgsl, permuteA } = await import('./kernel.wgsl.js?v=' + Date.now());
    const mod = await import('/engine.dflash.js?v=' + Date.now());
    const Eng = mod.TernaryBonsai2; const I = Eng.__dflashInternals; mark('imported');
    const m = await Eng.load('/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf', { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    mark('loaded');
    const inner = m.model, rt = inner.runtime, dev = rt.host.device;   // rt.device is the engine's normalized view; rt.host.device is the GPUDevice
    const H = inner.config.hidden_size, F = inner.config.intermediate_size, VOC = inner.config.vocab_size;
    const pack = inner.packs.lut2_32[0];
    const offUp = pack.offsets.layers[0].up_proj, offGate = pack.offsets.layers[0].gate_proj, offDown = pack.offsets.layers[0].down_proj;
    if (offUp % 8 !== 0 || offGate % 8 !== 0) throw new Error('block offsets not superblock aligned');
    const R = { layout: { H, F, VOC, lut: pack.lutIds.get('layers.0.up_proj'), offUp, offGate, offDown, bitsWords: pack.bits.shape[0], scalesWords: pack.scales.shape[0], lmHeadLut: inner.lmHeadLut, lmHeadBitsWords: inner.lmHeadQ4.shape[0], lmHeadScalesWords: inner.lmHeadQ4Scales.shape[0], adapter: dev.adapterInfo.architecture, features: [...dev.features].filter(f => /f16|subgroup|timestamp/.test(f)) } };
    V.results = R;

    // ---------- micro-graph helper on the engine's own compile/dispatch stack ----------
    class Micro extends I.f0 { constructor(emit) { super(inner, null, 8); this._emit = emit; } buildEmission() { return this._emit(); } }
    const micro = async (name, emit) => { const s = new Micro(() => { const S = new I.ba(); const b = I._i(S); const t = emit(S, b); return { graph: S.finish({ name }), weights: b.boundWeights, states: b.states, ...t }; }); await s.build(); return s; };
    const write = (s, name, data) => { const t = s.compiled.tensor(name); rt.host.writeBuffer(t.buffer, t.byteOffset ?? 0, data); };
    const WARM = Number(q.get('warm') || 60);
    const gpuMs = async (s) => { for (let i = 0; i < WARM; ++i) s.compiled.collector.enqueue(s.steps); await rt.queueIdle(); const r = await rt.measurePreparedSequenceGpuSamples(s.steps, 1, ITERS); return r; };
    const wallMs = async (s, n = ITERS) => { await rt.queueIdle(); const t0 = performance.now(); for (let i = 0; i < n; ++i) s.compiled.collector.enqueue(s.steps); await rt.queueIdle(); return (performance.now() - t0) / n; };

    // ---------- activation: 8 gaussian rows, rotated by the engine's own BlockHadamard (Prism signs for width H) ----------
    const X = new Float32Array(8 * H); for (let r = 0; r < 8; ++r) X.set(randn(H, 1000 + r), r * H);
    const rotS = await micro('spike-rotate', (S, b) => { const x = S.stepInput('x', 'float32', [8, H]); const y = I.bi(S, inner, b.w)(x, 'layers.0.up_proj'); if (y === x) throw new Error('no hadamard applied'); S.output(y, 'rot'); return {}; });
    write(rotS, 'x', X); rotS.compiled.collector.enqueue(rotS.steps);
    const A = await rt.readTensor(rotS.compiled.tensor('rot'));
    R.activation = { rows: 8, width: H, rotated: true, rms: fmt(Math.sqrt(A.reduce((s, v) => s + v * v, 0) / A.length)), sample: Array.from(A.slice(0, 4)).map(fmt) };
    mark('activation rotated');

    // ---------- engine reference 1: decode gate/up op (2 matrices of F x H), 1 token ----------
    const guS = await micro('spike-gateup', (S, b) => {
      const K = I.fi(b.w, inner)(0, ['gate_proj', 'up_proj'], 'gateup');
      const normed = S.stepInput('normed', 'float32', [H]);
      const inter = S.scratch('inter', 'float32', [F]);
      S.op('com.xenova.LlamaDecodeGateUp', { normedT: normed, bitsT: K.bitsT, scalesT: K.scalesT, intermediateT: inter }, { args: { hiddenSize: H, intermediateSize: F, gateOffset: K.offset('layers.0.gate_proj'), upOffset: K.offset('layers.0.up_proj'), format: K.format, lut: K.lut } });
      S.output(inter, 'out'); return {};
    });
    write(guS, 'normed', A.slice(0, H));
    R.engineDecodeGateUp = { steps: guS.steps.length, gpuMs: await gpuMs(guS), wallMs: fmt(await wallMs(guS)) };
    R.engineDecodeGateUp.gpuMedian = R.engineDecodeGateUp.gpuMs && fmt(median(R.engineDecodeGateUp.gpuMs));
    mark(`decode gate/up: ${JSON.stringify({ gpu: R.engineDecodeGateUp.gpuMedian, wall: R.engineDecodeGateUp.wallMs })}`);

    // ---------- engine reference 1b: decode gate/up op with tokens=4 (its pre-normalized multi-row mode) ----------
    try {
      const gu4 = await micro('spike-gateup4', (S, b) => {
        const K = I.fi(b.w, inner)(0, ['gate_proj', 'up_proj'], 'gateup');
        const normed = S.stepInput('normed', 'float32', [4 * H]);
        const inter = S.scratch('inter', 'float32', [4 * F]);
        S.op('com.xenova.LlamaDecodeGateUp', { normedT: normed, bitsT: K.bitsT, scalesT: K.scalesT, intermediateT: inter }, { args: { hiddenSize: H, intermediateSize: F, gateOffset: K.offset('layers.0.gate_proj'), upOffset: K.offset('layers.0.up_proj'), format: K.format, lut: K.lut, tokens: 4 } });
        S.output(inter, 'out'); return {};
      });
      write(gu4, 'normed', A.slice(0, 4 * H));
      const g = await gpuMs(gu4);
      R.engineDecodeGateUpTokens4 = { steps: gu4.steps.length, gpuMedian: g && fmt(median(g)), wallMs: fmt(await wallMs(gu4)) };
      gu4.dispose();
    } catch (e) { R.engineDecodeGateUpTokens4 = { error: String(e.message || e).slice(0, 300) }; }

    // ---------- engine reference 2: decode residual projection on down_proj (1 matrix H x F), 1 token ----------
    try {
      const dnS = await micro('spike-down', (S, b) => {
        const K = I.fi(b.w, inner)(0, ['down_proj'], 'down');
        const hid = S.stepInput('hidden', 'float32', [H]);
        const inter = S.stepInput('inter', 'float32', [F]);
        const out = S.op('com.xenova.LlamaDecodeResidualProjection', { hiddenT: hid, bitsT: K.bitsT, scalesT: K.scalesT, intermediateT: inter }, { args: { hiddenSize: H, intermediateSize: F, downOffset: K.offset('layers.0.down_proj'), format: K.format, lut: K.lut } }).hiddenT;
        S.output(out, 'out'); return {};
      });
      write(dnS, 'out', new Float32Array(H)); write(dnS, 'inter', randn(F, 7));   // hiddenT is renamed to 'out' by S.output (in-place op)
      const g = await gpuMs(dnS);
      R.engineDecodeDown = { steps: dnS.steps.length, gpuMedian: g && fmt(median(g)), wallMs: fmt(await wallMs(dnS)) };
      dnS.dispose();
    } catch (e) { R.engineDecodeDown = { error: String(e.message || e).slice(0, 300) }; }
    mark('decode refs measured');

    // ---------- engine reference 3: prefill matmul, M=8, up_proj alone ----------
    const pfS = await micro('spike-prefill-up', (S, b) => {
      const K = I.fi(b.w, inner)(0, ['up_proj'], 'up');
      const a = S.stepInput('a', 'float32', [8, H]);
      const y = S.scratch('y', 'float32', [8, F]);
      S.op('com.xenova.LlamaPrefillMatmul', { aT: a, bitsT: K.bitsT, scalesT: K.scalesT, yT: y }, { args: { M: 8, inFeatures: H, outFeatures: F, blockOffset: K.offset('layers.0.up_proj'), outStride: F, dstColStart: 0, format: K.format, lut: K.lut } });
      S.output(y, 'out'); return {};
    });
    write(pfS, 'a', A); pfS.compiled.collector.enqueue(pfS.steps);
    const Ypf = await rt.readTensor(pfS.compiled.tensor('out'));
    const gpf = await gpuMs(pfS);
    R.enginePrefillUp8 = { steps: pfS.steps.length, gpuMs: gpf, gpuMedian: gpf && fmt(median(gpf)), wallMs: fmt(await wallMs(pfS)) };
    mark(`prefill up M=8: ${R.enginePrefillUp8.gpuMedian} ms`);

    // ---------- engine reference 4: lm_head via LlamaPrefillMatmul at M=1 (the decode graph's head route) and M=8 ----------
    R.engineLmHead = {};
    for (const MM of [1, 8]) {
      try {
        const hs = await micro(`spike-head-${MM}`, (S, b) => {
          const bitsT = b.w('head.bits', inner.lmHeadQ4), scalesT = b.w('head.scales', inner.lmHeadQ4Scales);
          const a = S.stepInput('a', 'float32', [MM, H]); const y = S.scratch('y', 'float32', [MM, VOC]);
          S.op('com.xenova.LlamaPrefillMatmul', { aT: a, bitsT, scalesT, yT: y }, { args: { M: MM, inFeatures: H, outFeatures: VOC, blockOffset: 0, outStride: VOC, dstColStart: 0, format: 'lut2_32', lut: inner.lmHeadLut } });
          S.output(y, 'out'); return {};
        });
        write(hs, 'a', A.slice(0, MM * H));
        const g = await gpuMs(hs);
        R.engineLmHead[`M${MM}`] = { steps: hs.steps.length, gpuMedian: g && fmt(median(g)), wallMs: fmt(await wallMs(hs)) };
        hs.dispose();
      } catch (e) { R.engineLmHead[`M${MM}`] = { error: String(e.message || e).slice(0, 300) }; }
    }
    mark(`lm_head engine: ${JSON.stringify(R.engineLmHead)}`);

    // ---------- spike kernel ----------
    const qs = dev.createQuerySet({ type: 'timestamp', count: 2 * ITERS });
    const qres = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const qread = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const Abuf = dev.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Abuf, 0, A);
    const Aperm = permuteA(A, 8, H); const AbufPerm = dev.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(AbufPerm, 0, Aperm);
    R.layout.packedDotFeature = navigator.gpu.wgslLanguageFeatures ? navigator.gpu.wgslLanguageFeatures.has('packed_4x8_integer_dot_product') : null;
    const mkParams = (N, K, off) => { const b = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(b, 0, new Uint32Array([N, K, 2 * off, off / 8])); return b; };
    const compileKernel = async (code) => {
      const mod = dev.createShaderModule({ code });
      const info = await mod.getCompilationInfo();
      const errs = info.messages.filter(x => x.type === 'error').map(x => `${x.lineNum}:${x.linePos} ${x.message}`);
      if (errs.length) throw new Error('WGSL: ' + errs.join(' | '));
      return dev.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    };
    const reducePipe = await compileKernel(reduceWgsl());
    const timeDispatch = async (record) => {
      // warm: enough dispatches to pull the GPU clock up before the timed pass
      { const e = dev.createCommandEncoder(); for (let i = 0; i < WARM; ++i) { const p = e.beginComputePass(); record(p); p.end(); } dev.queue.submit([e.finish()]); await dev.queue.onSubmittedWorkDone(); }
      const t0 = performance.now();
      const enc = dev.createCommandEncoder();
      for (let i = 0; i < ITERS; ++i) { const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }); record(p); p.end(); }
      enc.resolveQuerySet(qs, 0, 2 * ITERS, qres, 0); enc.copyBufferToBuffer(qres, 0, qread, 0, 16 * ITERS);
      dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone();
      const wall = (performance.now() - t0) / ITERS;
      await qread.mapAsync(GPUMapMode.READ); const ts = new BigInt64Array(qread.getMappedRange().slice(0)); qread.unmap();
      const ms = []; for (let i = 0; i < ITERS; ++i) ms.push(Number(ts[2 * i + 1] - ts[2 * i]) / 1e6);
      return { ms, wall };
    };
    const readY = async (Ybuf, bytes) => { const stg = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(Ybuf, 0, stg, 0, bytes); dev.queue.submit([e.finish()]); await stg.mapAsync(GPUMapMode.READ); const Y = new Float32Array(stg.getMappedRange().slice(0)); stg.unmap(); stg.destroy(); return Y; };
    const runKernel = async (cfg, { bits, scales, N, K, off, label }) => {
      const Ybuf = dev.createBuffer({ size: 8 * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const TN = cfg.TN ?? 64; const nWG = Math.ceil(N / TN);
      let result;
      if (cfg.v3) {
        const pipe = await compileKernel(gemmWgslV3(cfg));
        const nChunks = K / (cfg.CH || (cfg.aStore === 'f16' ? 1024 : 512)); const cpw = cfg.chunksPerWG || nChunks; if (nChunks % cpw) throw new Error('chunksPerWG must divide K/1024');
        const nSplits = nChunks / cpw; const nParts = nSplits * 8;
        const Ypart = dev.createBuffer({ size: nParts * 8 * N * 4, usage: GPUBufferUsage.STORAGE });
        const params = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(params, 0, new Uint32Array([N, K, 2 * off, off / 8, cpw, nSplits, 0, 0]));
        const rparams = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(rparams, 0, new Uint32Array([N, nParts, 0, 0]));
        const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bits } }, { binding: 1, resource: { buffer: scales } }, { binding: 2, resource: { buffer: Abuf } }, { binding: 3, resource: { buffer: Ypart } }, { binding: 4, resource: { buffer: params } }] });
        const rbg = dev.createBindGroup({ layout: reducePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: Ypart } }, { binding: 1, resource: { buffer: Ybuf } }, { binding: 2, resource: { buffer: rparams } }] });
        const record = (p) => { p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(nWG, nSplits); p.setPipeline(reducePipe); p.setBindGroup(0, rbg); p.dispatchWorkgroups(Math.ceil(8 * N / 256)); };
        const { ms, wall } = await timeDispatch(record);
        const Y = await readY(Ybuf, 8 * N * 4);
        result = { label, cfg, nWG, nSplits, gpuMs: ms.map(fmt), gpuMedian: fmt(median(ms)), gpuMin: fmt(Math.min(...ms)), wallMsPerIter: fmt(wall), Y };
        Ypart.destroy(); params.destroy(); rparams.destroy();
      } else {
        const pipe = await compileKernel(cfg.v6 ? gemmWgslV6(cfg) : cfg.v5 ? gemmWgslV5(cfg) : cfg.v4 ? gemmWgslV4(cfg) : cfg.v2 ? gemmWgslV2(cfg) : gemmWgsl(cfg));
        const params = mkParams(N, K, off);
        const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: bits } }, { binding: 1, resource: { buffer: scales } }, { binding: 2, resource: { buffer: cfg.v4 ? AbufPerm : Abuf } }, { binding: 3, resource: { buffer: Ybuf } }, { binding: 4, resource: { buffer: params } }] });
        const record = (p) => { p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(nWG); };
        const { ms, wall } = await timeDispatch(record);
        const Y = await readY(Ybuf, 8 * N * 4);
        result = { label, cfg, nWG, gpuMs: ms.map(fmt), gpuMedian: fmt(median(ms)), gpuMin: fmt(Math.min(...ms)), wallMsPerIter: fmt(wall), Y };
        params.destroy();
      }
      Ybuf.destroy();
      return result;
    };

    // ---------- CPU reference for up_proj (float64 from dequantized weights) ----------
    const readRange = async (buf, byteOff, byteLen) => { const s = dev.createBuffer({ size: byteLen, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, byteOff, s, 0, byteLen); dev.queue.submit([e.finish()]); await s.mapAsync(GPUMapMode.READ); const out = new Uint32Array(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return out; };
    const cpuRef = (bitsW, scalesW, N, K, rows) => {
      // bitsW/scalesW start at the tensor's own block 0; rows = number of output rows to compute
      const BPR = K / 32; const Y = new Float64Array(8 * N);
      const w = new Float64Array(K);
      for (let n = 0; n < rows; ++n) {
        for (let kb = 0; kb < BPR; ++kb) {
          const blk = n * BPR + kb; const sw = scalesW[blk >> 3]; const s = f16ToF32(((blk >> 2) & 1) ? (sw >>> 16) : (sw & 0xffff));
          const wa = bitsW[2 * blk], wb = bitsW[2 * blk + 1];
          for (let e = 0; e < 16; ++e) { w[kb * 32 + e] = (((wa >>> (2 * e)) & 3) - 1) * s; w[kb * 32 + 16 + e] = (((wb >>> (2 * e)) & 3) - 1) * s; }
        }
        for (let mm = 0; mm < 8; ++mm) { let acc = 0; const ao = mm * K; for (let k = 0; k < K; ++k) acc += w[k] * A[ao + k]; Y[mm * N + n] = acc; }
      }
      return Y;
    };
    const compare = (Y, ref, N, rows) => {
      let maxAbs = 0, sumAbs = 0, maxRef = 0, cnt = 0, argmaxOk = 0, argRows = 0; let maxAbsAt = null;
      for (let mm = 0; mm < 8; ++mm) {
        let bi = -1, bv = -Infinity, ri = -1, rv = -Infinity;
        for (let n = 0; n < rows; ++n) { const y = Y[mm * N + n], r = ref[mm * N + n]; const d = Math.abs(y - r); if (d > maxAbs) { maxAbs = d; maxAbsAt = [mm, n, y, r]; } sumAbs += d; cnt++; maxRef = Math.max(maxRef, Math.abs(r)); if (y > bv) { bv = y; bi = n; } if (r > rv) { rv = r; ri = n; } }
        argRows++; if (bi === ri) argmaxOk++;
      }
      return { maxAbs: fmt(maxAbs), meanAbs: Number((sumAbs / cnt).toExponential(3)), maxRef: fmt(maxRef), maxAbsOverMaxRef: Number((maxAbs / maxRef).toExponential(3)), argmaxMatch: `${argmaxOk}/${argRows}`, worst: maxAbsAt && { m: maxAbsAt[0], n: maxAbsAt[1], got: fmt(maxAbsAt[2]), ref: fmt(maxAbsAt[3]) } };
    };
    mark('reading back up_proj words for the CPU reference');
    const upBits = await readRange(pack.bits.buffer, 2 * offUp * 4, F * (H / 32) * 2 * 4);
    const upScales = await readRange(pack.scales.buffer, (offUp / 8) * 4, F * (H / 32) / 8 * 4);
    const tRef = performance.now(); const refUp = cpuRef(upBits, upScales, F, H, F); R.cpuRefMs = fmt(performance.now() - tRef);
    // trit histogram (sanity: ternary codes only)
    { const hist = [0, 0, 0, 0]; for (let i = 0; i < upBits.length; i += 97) { const w = upBits[i]; for (let e = 0; e < 16; ++e) hist[(w >>> (2 * e)) & 3]++; } R.layout.tritCodeHistogramSampled = hist; }
    R.enginePrefillUp8.vsCpu = compare(Ypf, refUp, F, F);
    mark(`cpu ref done in ${R.cpuRefMs} ms; prefill vs cpu ${JSON.stringify(R.enginePrefillUp8.vsCpu)}`);

    // ---------- run spike kernel variants on up_proj ----------
    const variants = (q.get('variants') || '64x4m,s64x4,s64x2,s128x4,s32x4,s64x8').split(',').map(v => {
      const v6 = /^s(\d+)x(\d+)$/.exec(v); if (v6) return { v6: true, TN: +v6[1], RN: +v6[2] };
      const v5 = /^p(\d+)x(\d+)$/.exec(v); if (v5) return { v5: true, TN: +v5[1], RN: +v5[2] };
      const v4 = /^u(\d+)x(\d+)([fh])([bm])([vd])$/.exec(v); if (v4) return { v4: true, TN: +v4[1], RN: +v4[2], math: v4[3] === 'h' ? 'f16' : 'f32', aStore: v4[3] === 'h' ? 'f16' : 'f32', unpack: v4[4] === 'b' ? 'builtin' : 'manual', vecAcc: v4[5] === 'v' };
      const v3 = /^w(\d+)x(\d+)([fh])(?:c(\d+))?(?:k(\d+))?$/.exec(v); if (v3) return { v3: true, TN: +v3[1], RN: +v3[2], aStore: v3[3] === 'h' ? 'f16' : 'f32', chunksPerWG: v3[4] ? +v3[4] : 0, CH: v3[5] ? +v3[5] : 0 };
      const v2 = /^v(\d+)x(\d+)([fs]?)$/.exec(v); if (v2) return { v2: true, TN: +v2[1], RN: +v2[2], math: v2[3] === 'f' ? 'f32' : 'f16', aStore: v2[3] === 'f' ? 'f32' : 'f16', ...(v2[3] === 's' ? { math: 'f32', aStore: 'f16' } : {}) };
      const mm = /^(\d+)x(\d+)([hm]?)$/.exec(v); return { TN: +mm[1], RN: +mm[2], useF16A: mm[3] === 'h', f16Math: mm[3] === 'm' }; });
    R.spike = [];
    for (const cfg of variants) {
      try {
        const r = await runKernel(cfg, { bits: pack.bits.buffer, scales: pack.scales.buffer, N: F, K: H, off: offUp, label: cfg.v6 ? `up_proj v6 TN=${cfg.TN} RN=${cfg.RN} f16 16B-loads` : cfg.v5 ? `up_proj v5 TN=${cfg.TN} RN=${cfg.RN} f16 vec4-partials` : cfg.v4 ? `up_proj v4 TN=${cfg.TN} RN=${cfg.RN} math=${cfg.math} unpack=${cfg.unpack} ${cfg.vecAcc ? 'vecAcc' : 'dotAcc'}` : cfg.v3 ? `up_proj v3 TN=${cfg.TN} RN=${cfg.RN} A=${cfg.aStore} CH=${cfg.CH || (cfg.aStore === 'f16' ? 1024 : 512)} cpw=${cfg.chunksPerWG || 'all'}` : cfg.v2 ? `up_proj v2 TN=${cfg.TN} RN=${cfg.RN} math=${cfg.math} A=${cfg.aStore}` : `up_proj TN=${cfg.TN} RN=${cfg.RN}${cfg.useF16A ? ' f16A' : ''}${cfg.f16Math ? ' f16math' : ''}` });
        r.vsCpu = compare(r.Y, refUp, F, F); delete r.Y;
        R.spike.push(r); mark(`${r.label}: ${r.gpuMedian} ms, ${JSON.stringify(r.vsCpu)}`);
      } catch (e) { R.spike.push({ cfg, error: String(e.message || e).slice(0, 500) }); mark(`variant failed: ${String(e.message || e).slice(0, 200)}`); }
    }
    // ---------- lm_head (VOC x H) with the best variant; CPU check on the first 2048 rows ----------
    try {
      const best = R.spike.filter(x => !x.error).sort((a, b) => a.gpuMedian - b.gpuMedian)[0];
      const r = await runKernel(best.cfg, { bits: inner.lmHeadQ4.buffer, scales: inner.lmHeadQ4Scales.buffer, N: VOC, K: H, off: 0, label: `lm_head ${best.label.replace('up_proj ', '')}` });
      const rows = 2048;
      const hb = await readRange(inner.lmHeadQ4.buffer, 0, rows * (H / 32) * 2 * 4); const hs = await readRange(inner.lmHeadQ4Scales.buffer, 0, rows * (H / 32) / 8 * 4);
      const refHead = cpuRef(hb, hs, VOC, H, rows);
      r.vsCpuFirst2048Rows = compare(r.Y, refHead, VOC, rows); delete r.Y;
      R.lmHead = r; mark(`lm_head: ${r.gpuMedian} ms`);
    } catch (e) { R.lmHead = { error: String(e.message || e).slice(0, 500) }; }

    // ---------- re-measure the decode gate/up op after the kernel runs (same GPU clock state) ----------
    { const g = await gpuMs(guS); R.engineDecodeGateUpAfter = { gpuMedian: g && fmt(median(g)) }; }
    // ---------- ratios ----------
    const best = R.spike.filter(x => !x.error).sort((a, b) => a.gpuMedian - b.gpuMedian)[0];
    const perMatrixDecode = Math.min(R.engineDecodeGateUp.gpuMedian, R.engineDecodeGateUpAfter.gpuMedian) / 2;
    R.summary = {
      bestVariant: best.label, best8RowMs: best.gpuMedian,
      decodeGateUpMs_2matrices: R.engineDecodeGateUp.gpuMedian, decodeGateUpMs_2matrices_after: R.engineDecodeGateUpAfter.gpuMedian, decodePerMatrixMs_gateUpHalf: fmt(perMatrixDecode), decodeDownMs_1matrix: R.engineDecodeDown.gpuMedian,
      prefillUp8Ms: R.enginePrefillUp8.gpuMedian,
      ratio_best_over_gateUpHalf: fmt(best.gpuMedian / perMatrixDecode),
      ratio_best_over_down: R.engineDecodeDown.gpuMedian ? fmt(best.gpuMedian / R.engineDecodeDown.gpuMedian) : null,
      ratio_prefill8_over_gateUpHalf: fmt(R.enginePrefillUp8.gpuMedian / perMatrixDecode),
      speedup_vs_prefill8: fmt(R.enginePrefillUp8.gpuMedian / best.gpuMedian),
      weightBytesUp: F * H * 2 / 8 + F * H / 128 * 2, effGBps_best: fmt((F * H * 2 / 8 + F * H / 128 * 2) / (best.gpuMedian / 1e3) / 1e9),
    };
    mark('done');
  } catch (e) { V.error = String(e && e.stack || e); mark('error'); }
})();
