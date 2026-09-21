// Follow-up bench: M = 4 / 5 (and 8) row GEMM variants on up_proj, down_proj and lm_head, each against the engine's own
// decode matvec for that projection, with a float64 CPU reference. URL params: ?ms=4,5&variants=...&iters=41&rows=2048
// Results in window.__gs.results; run with cdp-drive.mjs.
window.__gs = { state: 'init', log: [] };
const V = window.__gs; const mark = (m) => { V.log.push(m); V.state = m; };
const q = new URLSearchParams(location.search);
const ITERS = Number(q.get('iters') || 41), WARM = Number(q.get('warm') || 60);
const MS = (q.get('ms') || '4,5').split(',').map(Number);
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const fmt = (x) => Number(x.toFixed(4));
function f16ToF32(h) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff; if (e === 0) return s * f * 2 ** -24; if (e === 31) return f ? NaN : s * Infinity; return s * (1 + f / 1024) * 2 ** (e - 15); }
function randn(n, seed) { let s = seed >>> 0; const out = new Float32Array(n); const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s + 0.5) / 4294967296; }; for (let i = 0; i < n; ++i) { const u = rnd(), v = rnd(); out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); } return out; }
// variant syntax: m<TN>x<RN><math f|h><tile f|h>[u][k<ksplit>]   e.g. m64x4hh = f16 math, f16 tile; m64x4fhu = f32 math, f16 tile, unrolled; k4 = 4 K-splits
const parseVariant = (v) => { const o = /^o(f16|f32|exact)(?:k(\d+))?$/.exec(v); if (o) return { op: true, precision: o[1], kSplits: o[2] ? +o[2] : 0 }; const r = /^m(\d+)x(\d+)([fh])([fh])(u?)(?:k(\d+))?(?:d([ws]))?$/.exec(v); if (!r) throw new Error('bad variant ' + v); return { TN: +r[1], RN: +r[2], math: r[3] === 'h' ? 'f16' : 'f32', aStore: r[4] === 'h' ? 'f16' : 'f32', unroll: r[5] === 'u', ksplit: r[6] ? +r[6] : 1, dequant: r[7] === 'w' ? 'lutw' : r[7] === 's' ? 'luts' : 'alu' }; };
const label = (cfg) => cfg.op ? `engine op Lut2SmallMGemm precision=${cfg.precision} kSplits=${cfg.kSplits || 'auto'}` : `TN=${cfg.TN} RN=${cfg.RN} math=${cfg.math} tile=${cfg.aStore}${cfg.unroll ? ' unrolled' : ''}${cfg.ksplit > 1 ? ` ksplit=${cfg.ksplit}` : ''}${cfg.dequant && cfg.dequant !== 'alu' ? ` dequant=${cfg.dequant}` : ''}`;
(async () => {
  try {
    const { gemmWgslM, reduceWgslM, lutTable } = await import('./kernel.wgsl.js?v=' + Date.now());
    const mod = await import('/engine.dflash.js?v=' + Date.now());
    const Eng = mod.TernaryBonsai2; const I = Eng.__dflashInternals; mark('imported');
    const m = await Eng.load('/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf', { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    mark('loaded');
    const inner = m.model, rt = inner.runtime, dev = rt.host.device;
    const H = inner.config.hidden_size, F = inner.config.intermediate_size, VOC = inner.config.vocab_size;
    const pack = inner.packs.lut2_32[0];
    const offUp = pack.offsets.layers[0].up_proj, offGate = pack.offsets.layers[0].gate_proj, offDown = pack.offsets.layers[0].down_proj;
    const R = { layout: { H, F, VOC, offUp, offGate, offDown }, iters: ITERS, warm: WARM, MS }; V.results = R;

    class Micro extends I.f0 { constructor(emit) { super(inner, null, 8); this._emit = emit; } buildEmission() { return this._emit(); } }
    const micro = async (name, emit) => { const s = new Micro(() => { const S = new I.ba(); const b = I._i(S); const t = emit(S, b); return { graph: S.finish({ name }), weights: b.boundWeights, states: b.states, ...t }; }); await s.build(); return s; };
    const write = (s, name, data) => { const t = s.compiled.tensor(name); rt.host.writeBuffer(t.buffer, t.byteOffset ?? 0, data); };
    const gpuMs = async (s) => { for (let i = 0; i < WARM; ++i) s.compiled.collector.enqueue(s.steps); await rt.queueIdle(); return rt.measurePreparedSequenceGpuSamples(s.steps, 1, ITERS); };
    const rotate = async (rows, K, weightName) => {
      const X = new Float32Array(rows * K); for (let r = 0; r < rows; ++r) X.set(randn(K, 1000 + r), r * K);
      const s = await micro(`rot-${weightName}-${rows}`, (S, b) => { const x = S.stepInput('x', 'float32', [rows, K]); const y = I.bi(S, inner, b.w)(x, weightName); if (y === x) throw new Error('no hadamard'); S.output(y, 'rot'); return {}; });
      write(s, 'x', X); s.compiled.collector.enqueue(s.steps); const A = await rt.readTensor(s.compiled.tensor('rot')); s.dispose(); return A;
    };

    // ---- engine decode references (1 token) ----
    R.engine = {};
    { const s = await micro('gateup', (S, b) => { const K = I.fi(b.w, inner)(0, ['gate_proj', 'up_proj'], 'gateup'); const n = S.stepInput('normed', 'float32', [H]); const o = S.scratch('inter', 'float32', [F]); S.op('com.xenova.LlamaDecodeGateUp', { normedT: n, bitsT: K.bitsT, scalesT: K.scalesT, intermediateT: o }, { args: { hiddenSize: H, intermediateSize: F, gateOffset: K.offset('layers.0.gate_proj'), upOffset: K.offset('layers.0.up_proj'), format: K.format, lut: K.lut } }); S.output(o, 'out'); return {}; });
      write(s, 'normed', randn(H, 3)); const g = await gpuMs(s); R.engine.gateUp2 = fmt(median(g)); R.engine.upPerMatrix = fmt(median(g) / 2); s.dispose(); }
    { const s = await micro('down', (S, b) => { const K = I.fi(b.w, inner)(0, ['down_proj'], 'down'); const hid = S.stepInput('hidden', 'float32', [H]); const inter = S.stepInput('inter', 'float32', [F]); const out = S.op('com.xenova.LlamaDecodeResidualProjection', { hiddenT: hid, bitsT: K.bitsT, scalesT: K.scalesT, intermediateT: inter }, { args: { hiddenSize: H, intermediateSize: F, downOffset: K.offset('layers.0.down_proj'), format: K.format, lut: K.lut } }).hiddenT; S.output(out, 'out'); return {}; });
      write(s, 'out', new Float32Array(H)); write(s, 'inter', randn(F, 7)); const g = await gpuMs(s); R.engine.down = fmt(median(g)); s.dispose(); }
    { const s = await micro('head1', (S, b) => { const bitsT = b.w('head.bits', inner.lmHeadQ4), scalesT = b.w('head.scales', inner.lmHeadQ4Scales); const a = S.stepInput('a', 'float32', [1, H]); const y = S.scratch('y', 'float32', [1, VOC]); S.op('com.xenova.LlamaPrefillMatmul', { aT: a, bitsT, scalesT, yT: y }, { args: { M: 1, inFeatures: H, outFeatures: VOC, blockOffset: 0, outStride: VOC, dstColStart: 0, format: 'lut2_32', lut: inner.lmHeadLut } }); S.output(y, 'out'); return {}; });
      write(s, 'a', randn(H, 5)); const g = await gpuMs(s); R.engine.head1 = fmt(median(g)); s.dispose(); }
    // engine prefill matmul at M = 4 / 5 (what the prefill graph would use today at those row counts)
    R.enginePrefill = {};
    for (const MM of MS) {
      const s = await micro(`pf-up-${MM}`, (S, b) => { const K = I.fi(b.w, inner)(0, ['up_proj'], 'up'); const a = S.stepInput('a', 'float32', [MM, H]); const y = S.scratch('y', 'float32', [MM, F]); S.op('com.xenova.LlamaPrefillMatmul', { aT: a, bitsT: K.bitsT, scalesT: K.scalesT, yT: y }, { args: { M: MM, inFeatures: H, outFeatures: F, blockOffset: K.offset('layers.0.up_proj'), outStride: F, dstColStart: 0, format: K.format, lut: K.lut } }); S.output(y, 'out'); return {}; });
      write(s, 'a', randn(MM * H, 11)); const g = await gpuMs(s); R.enginePrefill[`up_M${MM}`] = fmt(median(g)); s.dispose();
    }
    mark(`engine refs ${JSON.stringify(R.engine)} prefill ${JSON.stringify(R.enginePrefill)}`);

    // ---- spike kernel machinery ----
    const qs = dev.createQuerySet({ type: 'timestamp', count: 2 * ITERS });
    const qres = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const qread = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipeCache = new Map();
    const compile = async (code) => { if (pipeCache.has(code)) return pipeCache.get(code); const mod = dev.createShaderModule({ code }); const info = await mod.getCompilationInfo(); const errs = info.messages.filter(x => x.type === 'error').map(x => `${x.lineNum}:${x.linePos} ${x.message}`); if (errs.length) throw new Error('WGSL: ' + errs.join(' | ')); const p = await dev.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } }); pipeCache.set(code, p); return p; };
    const timeDispatch = async (record) => {
      { const e = dev.createCommandEncoder(); for (let i = 0; i < WARM; ++i) { const p = e.beginComputePass(); record(p); p.end(); } dev.queue.submit([e.finish()]); await dev.queue.onSubmittedWorkDone(); }
      const t0 = performance.now(); const enc = dev.createCommandEncoder();
      for (let i = 0; i < ITERS; ++i) { const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } }); record(p); p.end(); }
      enc.resolveQuerySet(qs, 0, 2 * ITERS, qres, 0); enc.copyBufferToBuffer(qres, 0, qread, 0, 16 * ITERS); dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone();
      const wall = (performance.now() - t0) / ITERS; await qread.mapAsync(GPUMapMode.READ); const ts = new BigInt64Array(qread.getMappedRange().slice(0)); qread.unmap();
      const ms = []; for (let i = 0; i < ITERS; ++i) ms.push(Number(ts[2 * i + 1] - ts[2 * i]) / 1e6); return { ms, wall };
    };
    const readBuf = async (buf, byteOff, byteLen, Ctor) => { const s = dev.createBuffer({ size: byteLen, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(buf, byteOff, s, 0, byteLen); dev.queue.submit([e.finish()]); await s.mapAsync(GPUMapMode.READ); const out = new Ctor(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return out; };
    const runOp = async (M, cfg, pj, A) => {
      const s = await micro(`op-${pj.name}-${M}-${cfg.precision}`, (S, b) => {
        const bitsT = b.w(`${pj.name}.bits`, pj.bitsTensor), scalesT = b.w(`${pj.name}.scales`, pj.scalesTensor);
        const a = S.stepInput('a', 'float32', [M, pj.K]); const y = S.scratch('y', 'float32', [M, pj.N]);
        S.op('com.xenova.Lut2SmallMGemm', { aT: a, bitsT, scalesT, yT: y }, { args: { M, inFeatures: pj.K, outFeatures: pj.N, blockOffset: pj.off, outStride: pj.N, dstColStart: 0, lut: 9, precision: cfg.precision, ...(cfg.kSplits ? { kSplits: cfg.kSplits } : {}) } });
        S.output(y, 'out'); return {};
      });
      write(s, 'a', A); s.compiled.collector.enqueue(s.steps); const Y = await rt.readTensor(s.compiled.tensor('out'));
      const g = await gpuMs(s); const wall = await wallMs(s); s.dispose();
      return { label: label(cfg), cfg, steps: s.steps.length, gpuMedian: fmt(median(g)), gpuMin: fmt(Math.min(...g)), wallMsPerIter: fmt(wall), Y };
    };
    const wallMs = async (s, n = ITERS) => { await rt.queueIdle(); const t0 = performance.now(); for (let i = 0; i < n; ++i) s.compiled.collector.enqueue(s.steps); await rt.queueIdle(); return (performance.now() - t0) / n; };
    const runKernel = async (M, cfg, { bits, scales, N, K, off, Abuf }) => {
      const pipe = await compile(gemmWgslM({ M, ...cfg }));
      const nWG = Math.ceil(N / cfg.TN), nChunks = K / 256; if (nChunks % cfg.ksplit) throw new Error('ksplit must divide K/256');
      const cps = nChunks / cfg.ksplit;
      const Ybuf = dev.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const Ypart = cfg.ksplit > 1 ? dev.createBuffer({ size: cfg.ksplit * M * N * 4, usage: GPUBufferUsage.STORAGE }) : Ybuf;
      const params = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(params, 0, new Uint32Array([N, K, 2 * off, off / 8, cps, 0, 0, 0]));
      let lutBuf = null; if (cfg.dequant === 'luts') { const t = lutTable(cfg.math); lutBuf = dev.createBuffer({ size: t.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(lutBuf, 0, t); }
      const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bits } }, { binding: 1, resource: { buffer: scales } }, { binding: 2, resource: { buffer: Abuf } }, { binding: 3, resource: { buffer: Ypart } }, { binding: 4, resource: { buffer: params } }, ...(lutBuf ? [{ binding: 5, resource: { buffer: lutBuf } }] : [])] });
      let rpipe = null, rbg = null, rparams = null;
      if (cfg.ksplit > 1) { rpipe = await compile(reduceWgslM(M)); rparams = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(rparams, 0, new Uint32Array([N, cfg.ksplit, 0, 0])); rbg = dev.createBindGroup({ layout: rpipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: Ypart } }, { binding: 1, resource: { buffer: Ybuf } }, { binding: 2, resource: { buffer: rparams } }] }); }
      const record = (p) => { p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(nWG, cfg.ksplit); if (rpipe) { p.setPipeline(rpipe); p.setBindGroup(0, rbg); p.dispatchWorkgroups(Math.ceil(M * N / 256)); } };
      const { ms, wall } = await timeDispatch(record);
      const Y = await readBuf(Ybuf, 0, M * N * 4, Float32Array);
      Ybuf.destroy(); if (Ypart !== Ybuf) Ypart.destroy(); params.destroy(); rparams?.destroy(); lutBuf?.destroy();
      return { label: label(cfg), cfg, nWG: nWG * cfg.ksplit, gpuMedian: fmt(median(ms)), gpuMin: fmt(Math.min(...ms)), wallMsPerIter: fmt(wall), Y };
    };
    const cpuRef = (bitsW, scalesW, M, N, K, rows, A) => {
      const BPR = K / 32; const Y = new Float64Array(M * N); const w = new Float64Array(K);
      for (let n = 0; n < rows; ++n) {
        for (let kb = 0; kb < BPR; ++kb) { const blk = n * BPR + kb; const sw = scalesW[blk >> 3]; const s = f16ToF32(((blk >> 2) & 1) ? (sw >>> 16) : (sw & 0xffff)); const wa = bitsW[2 * blk], wb = bitsW[2 * blk + 1]; for (let e = 0; e < 16; ++e) { w[kb * 32 + e] = (((wa >>> (2 * e)) & 3) - 1) * s; w[kb * 32 + 16 + e] = (((wb >>> (2 * e)) & 3) - 1) * s; } }
        for (let mm = 0; mm < M; ++mm) { let acc = 0; const ao = mm * K; for (let k = 0; k < K; ++k) acc += w[k] * A[ao + k]; Y[mm * N + n] = acc; }
      }
      return Y;
    };
    const compare = (Y, ref, M, N, rows) => {
      let maxAbs = 0, sumAbs = 0, maxRef = 0, cnt = 0, argmaxOk = 0, worst = null;
      for (let mm = 0; mm < M; ++mm) { let bi = -1, bv = -Infinity, ri = -1, rv = -Infinity; for (let n = 0; n < rows; ++n) { const y = Y[mm * N + n], r = ref[mm * N + n]; const d = Math.abs(y - r); if (d > maxAbs) { maxAbs = d; worst = { m: mm, n, got: fmt(y), ref: fmt(r) }; } sumAbs += d; cnt++; maxRef = Math.max(maxRef, Math.abs(r)); if (y > bv) { bv = y; bi = n; } if (r > rv) { rv = r; ri = n; } } if (bi === ri) argmaxOk++; }
      return { maxAbs: Number(maxAbs.toExponential(2)), meanAbs: Number((sumAbs / cnt).toExponential(2)), maxRef: fmt(maxRef), maxAbsOverMaxRef: Number((maxAbs / maxRef).toExponential(2)), argmaxMatch: `${argmaxOk}/${M}`, worst };
    };

    // ---- projections ----
    const headRows = Number(q.get('rows') || 2048);
    const projs = [
      { name: 'up_proj', bitsTensor: pack.bits, scalesTensor: pack.scales, bits: pack.bits.buffer, scales: pack.scales.buffer, N: F, K: H, off: offUp, rot: 'layers.0.up_proj', decodeMs: R.engine.upPerMatrix, refRows: F },
      { name: 'down_proj', bitsTensor: pack.bits, scalesTensor: pack.scales, bits: pack.bits.buffer, scales: pack.scales.buffer, N: H, K: F, off: offDown, rot: 'layers.0.down_proj', decodeMs: R.engine.down, refRows: H },
      { name: 'lm_head', bitsTensor: inner.lmHeadQ4, scalesTensor: inner.lmHeadQ4Scales, bits: inner.lmHeadQ4.buffer, scales: inner.lmHeadQ4Scales.buffer, N: VOC, K: H, off: 0, rot: 'layers.0.up_proj', decodeMs: R.engine.head1, refRows: headRows },
    ].filter(p => (q.get('projs') || 'up_proj,down_proj,lm_head').split(',').includes(p.name));
    const defaultVariants = { up_proj: 'm64x4hhu,m64x4hh,m64x4fhu,m64x4ffu', down_proj: 'm64x4hhuk4,m64x4hhk4,m64x4fhuk4,m64x4ffuk4,m64x4hhu', lm_head: 'm64x4hhu,m64x8hh,m64x4hh,m64x4fhu,m64x4ffu' };   // the sweep winners (m45.log / m45b.log / m45c.log hold the full sweeps)
    R.runs = {};
    for (const M of MS) {
      for (const pj of projs) {
        const A = await rotate(M, pj.K, pj.rot);
        const Abuf = dev.createBuffer({ size: A.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Abuf, 0, A);
        const bitsW = await readBuf(pj.bits, 2 * pj.off * 4, pj.refRows * (pj.K / 32) * 2 * 4, Uint32Array);
        const scalesW = await readBuf(pj.scales, (pj.off / 8) * 4, pj.refRows * (pj.K / 32) / 8 * 4, Uint32Array);
        const t0 = performance.now(); const ref = cpuRef(bitsW, scalesW, M, pj.N, pj.K, pj.refRows, A); const refMs = fmt(performance.now() - t0);
        const variants = (q.get('variants') || defaultVariants[pj.name]).split(',').map(parseVariant);
        const key = `M${M}/${pj.name}`; R.runs[key] = { decodeMs: pj.decodeMs, refRows: pj.refRows, refMs, variants: [] };
        for (const cfg of variants) {
          try {
            const r = cfg.op ? await runOp(M, cfg, pj, A) : await runKernel(M, cfg, { bits: pj.bits, scales: pj.scales, N: pj.N, K: pj.K, off: pj.off, Abuf });
            r.vsCpu = compare(r.Y, ref, M, pj.N, pj.refRows); delete r.Y; r.ratio = fmt(r.gpuMedian / pj.decodeMs);
            R.runs[key].variants.push(r); mark(`${key} ${r.label}: ${r.gpuMedian} ms ratio ${r.ratio} maxAbs ${r.vsCpu.maxAbs} argmax ${r.vsCpu.argmaxMatch}`);
          } catch (e) { R.runs[key].variants.push({ label: label(cfg), cfg, error: String(e.message || e).slice(0, 400) }); mark(`${key} ${label(cfg)} FAILED ${String(e.message || e).slice(0, 120)}`); }
        }
        Abuf.destroy();
        const ok = R.runs[key].variants.filter(v => !v.error);
        R.runs[key].best = ok.length ? ok.reduce((a, b) => a.gpuMedian < b.gpuMedian ? a : b).label : null;
        R.runs[key].bestRatio = ok.length ? Math.min(...ok.map(v => v.ratio)) : null;
      }
    }
    mark('done');
  } catch (e) { V.error = String(e && e.stack || e); mark('error'); }
})();
