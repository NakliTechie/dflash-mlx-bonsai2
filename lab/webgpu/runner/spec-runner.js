// DFlashRunner — the DFlash 2 speculative generate loop behind the engine's `specDecodeRunner()` seam.
// The harness in runner.js measures; this file is the reusable class an app attaches to the engine:
//
//   const runner = await DFlashRunner.create(engine, { drafter: urlOrByteSource, block: 5, onProgress });
//   engine.model.dflashRunner = runner;          // the patched engine routes generate() through it
//   ...
//   runner.dispose();
//
// Engine contract (ternary_bonsai_2_27b.js after scripts/bonsai2-dflash-patches.mjs): the graph-decode path calls
// `runner.generate(suffixIds, cache, generationArgs, beginDecode, eosTokenId)` and consumes it as the token stream;
// `runner.release(cache)` runs before a temporary cache is disposed. Invariant kept from the engine's own loop: at
// every yield the cache holds the prompt plus every yielded token except the last one (the wrapper commits its prefix
// cache from `cache.get_seq_length()`, so the cache content must equal the first seqLength tokens of prompt+output).
//
// Cycle: capture checkpoint -> draft 7 (noise rows through the drafter, lm_head micro graph, GPU top-16, selector)
//   -> verify block [anchor, drafts[0..block-2]] through the tapped small-M verify session -> accept the longest
//   agreeing prefix (k) + the bonus token -> append rows 0..k of the verify run's features to the drafter context
//   -> if k+1 < block: restore the checkpoint and rewind the recurrent state through the accepted rows (RewindSession).
// Optional prompt lookup (`ngram: true`, `ngramK`): when the context's suffix (>= ngramK tokens) occurred earlier in this
// cache's token history, the tokens that followed it are the draft (the drafter runs only to fill a short match). The
// verify block, acceptance, context append and rewind are the same, so output and drafter context are unchanged.
// Output is greedy-identical to the engine's own decode (runner/RESULTS.md).
import { Drafter, CFG } from '../drafter/drafter.js';

const TAPS = [6, 20, 34, 48, 62];   // entry-of-layer convention == output of the drafter's target_layer_ids [5,19,33,47,61]
// Prompt prefill mirrors the engine's own chunk plan (graph blocks 16/32/64/128/256/512, the smallest that holds the
// remainder) through tapped sessions, so it costs what the engine's prefill costs; only the rows inside the drafter's
// context window (sink + window, the tail of the prompt) are appended to the drafter context.
const PREFILL_BLOCKS = [16, 32, 64, 128, 256, 512];

export class DFlashRunner {
  static async create(engine, opts = {}) {
    const r = new DFlashRunner(engine, opts);
    await r.#init(opts);
    // build the per-cache sessions for the engine's persistent cache now, off the first turn's critical path
    const cache = engine.generationState?.cache;
    if (cache) await r.#resources(cache);
    return r;
  }

  constructor(engine, opts) {
    this.engine = engine; this.inner = engine.model; this.rt = this.inner.runtime; this.dev = this.rt.host.device; this.cfg = this.inner.config;
    this.I = engine.constructor.__dflashInternals;
    if (!this.I || !this.I.lh || !this.I.ch || !this.I.RewindSession) throw new Error('DFlashRunner: engine is missing the dflash internals hook (regenerate ternary_bonsai_2_27b.js)');
    this.block = opts.block ?? 5;
    if (!(this.block >= 2 && this.block <= CFG.BLOCK)) throw new Error(`DFlashRunner: block must be 2..${CFG.BLOCK}`);
    this.sink = opts.sink ?? 64; this.window = opts.window ?? 1024;
    this.smallM = opts.smallM === 'off' ? undefined : { precision: opts.smallM ?? 'f16' };
    this.ngram = !!opts.ngram; this.ngramK = opts.ngramK ?? 3;
    if (!(this.ngramK >= 1)) throw new Error('DFlashRunner: ngramK must be >= 1');
    this.log = opts.log || (() => {});
    this.H = this.cfg.hidden_size; this.VOC = this.cfg.vocab_size;
    if (this.H !== CFG.H || this.VOC !== CFG.VOCAB) throw new Error(`DFlashRunner: target H/V ${this.H}/${this.VOC} != drafter ${CFG.H}/${CFG.VOCAB}`);
    this.res = new Map();      // cache -> { sessions, rewind, slot, ctx, ctxEnd, hist }
    this.stats = { cycles: 0, tokens: 0, accepted: 0, generations: 0, ngramCycles: 0, ngramAccepted: 0 };
    this.disposed = false;
  }

  async #init(opts) {
    const { I, inner, rt, dev, H, VOC } = this;
    // drafter on the engine's device
    this.dr = new Drafter(dev, { log: this.log, packed: opts.packed !== false });
    await this.dr.loadWeights(opts.drafter, opts.onProgress || (() => {}));
    // micro graphs on the engine's compile stack (pattern: wgsl-gemm-spike/bench.js)
    class Micro extends I.f0 { constructor(emit) { super(inner, null, 8); this._emit = emit; } buildEmission() { return this._emit(); } }
    const micro = async (name, emit) => { const s = new Micro(() => { const S = new I.ba(); const b = I._i(S); const t = emit(S, b); return { graph: S.finish({ name }), weights: b.boundWeights, states: b.states, ...t }; }); await s.build(); return s; };
    // (a) noise embedding: LlamaEmbed + inverse Hadamard, the same ops the prefill prologue emits
    this.embedS = await micro('dflash-embed', (S, b) => {
      const re = b.w; const ids = S.stepInput('input_ids', 'uint32', [8]); const hid = S.scratch('embed.hidden', 'float32', [8, H]); let h;
      if (inner.embedBits && inner.embedScales) h = S.op('com.xenova.LlamaEmbed', { inputT: ids, bitsT: re('embed.bits', inner.embedBits), scalesT: re('embed.scales', inner.embedScales), hiddenT: hid }, { args: { hiddenSize: H, vocabSize: VOC, seqLen: 8, format: inner.embedFormat ?? 'q8_rows' } }).hiddenT;
      else { const ke = I.pi(re, inner)('top'); h = S.op('com.xenova.LlamaEmbed', { inputT: ids, weightsT: ke, hiddenT: hid }, { args: { embedOffset: inner.offsets.top.embed_tokens, hiddenSize: H, vocabSize: VOC, seqLen: 8 } }).hiddenT; }
      h = I.bi(S, inner, re)(h, 'embed_tokens', true); S.output(h, 'embed'); return {};
    });
    const embedT = this.embedS.compiled.tensor('embed'); this.embedBind = { buffer: embedT.buffer, offset: embedT.byteOffset ?? 0, size: 8 * H * 4 };
    // (b) head: Hadamard rotation of the drafter's final hidden + Lut2SmallMGemm (M = 7) over the lm_head pack
    if (!(inner.lmHeadQ4 && inner.lmHeadQ4Scales && inner.lmHeadLut === 9 && this.cfg.prismHadamard?.weights.includes('output.weight'))) throw new Error('DFlashRunner: expected a prism-rotated lut2 lm_head (lmHeadLut 9)');
    const R = CFG.BLOCK - 1; this.R = R;
    const prec = (this.smallM && this.smallM.precision) || 'f16';
    this.headS = await micro('dflash-head', (S, b) => {
      const re = b.w; const x = S.stepInput('hid', 'float32', [R, H]); const rot = I.bi(S, inner, re)(x, 'lm_head'); const lg = S.scratch('logits', 'float32', [R, VOC]);
      S.op('com.xenova.Lut2SmallMGemm', { aT: S.view(rot, 0, 'float32', [R, H], 'head.in'), bitsT: re('head.bits', inner.lmHeadQ4), scalesT: re('head.scales', inner.lmHeadQ4Scales), yT: lg }, { args: { M: R, inFeatures: H, outFeatures: VOC, blockOffset: 0, outStride: VOC, dstColStart: 0, lut: 9, precision: prec, kSplits: 1 } });
      S.output(lg, 'logits'); return {};
    });
    this.headIn = this.headS.compiled.tensor('hid'); const headLg = this.headS.compiled.tensor('logits'); this.headLgBind = { buffer: headLg.buffer, offset: headLg.byteOffset ?? 0, size: R * VOC * 4 };
    this.log(`DFlashRunner ready: block ${this.block}, drafter ${(this.dr.stats.weightBytes / 1e9).toFixed(2)} GB on GPU`);
  }

  // ---- per-cache resources ----
  async #resources(cache) {
    let r = this.res.get(cache); if (r) return r;
    const { I, inner } = this; const taps = TAPS, smallM = this.smallM, LV = this.block;
    const sessions = new Map();
    // verify: all-rows head on the small-M route + the recurrence tee; prefill: taps only (next_token from the last row)
    const session = async (T, opts) => { let s = sessions.get(T); if (s) return s; class S extends I.ch { buildEmission() { return I.lh(this.model, this.cache, this.blockLen, opts); } } s = new S(inner, cache, T); await s.build(); sessions.set(T, s); return s; };
    const verify = await session(LV, { tapLayers: taps, allRowsHead: true, ...(smallM ? { smallM } : {}), teeRecurrence: true });
    const prefill = (T) => session(T, { tapLayers: taps });
    await prefill(16); await prefill(512);   // the two blocks every turn tends to need: a short suffix, a long first prompt
    const rewind = new I.RewindSession(inner, cache, LV, verify); await rewind.build();
    const slot = await cache.allocateCheckpointSlot();
    const ctx = this.dr.createContext(this.sink + this.window > 0 ? this.sink + this.window + 16 : cache.maxLength + 16, { sink: this.sink, window: this.window });
    // hist: the token ids the cache holds (prompt lookup)
    r = { sessions, verify, prefill, rewind, slot, ctx, ctxEnd: -1, hist: [] }; this.res.set(cache, r); return r;
  }

  release(cache) {
    const r = this.res.get(cache); if (!r) return; this.res.delete(cache);
    for (const s of r.sessions.values()) { try { s.dispose(); } catch (_) {} }
    try { r.rewind.dispose(); } catch (_) {}
    try { this.I.r2(r.slot.storage); } catch (_) {}
    try { this.dr.destroyContext?.(r.ctx); } catch (_) {}
  }

  dispose() {
    if (this.disposed) return; this.disposed = true;
    for (const cache of [...this.res.keys()]) this.release(cache);
    try { this.embedS.dispose(); } catch (_) {}
    try { this.headS.dispose(); } catch (_) {}
    try { this.dr.dispose?.(); } catch (_) {}
  }

  // run `ids` (1..T) through a tapped session at past_len = cache.seqLength; advances the cache
  async #runBlock(r, s, ids) {
    const { rt, H } = this; const T = ids.length; const pos = r.cache.seqLength;
    const next = await s.run(new Uint32Array(ids), pos); r.cache.seqLength = pos + T;
    const ft = s.compiled.tensor('dspark.features');
    return { next, pos, feat: { buffer: ft.buffer, offset: ft.byteOffset ?? 0, size: T * 5 * H * 4 }, tokens: () => rt.readTensor(s.compiled.tensor('verify_tokens')) };
  }

  #embed(ids) { const { rt } = this; rt.host.writeBuffer(this.embedS.compiled.tensor('input_ids').buffer, 0, new Uint32Array(ids)); this.embedS.compiled.collector.enqueue(this.embedS.steps); return this.embedBind; }

  async #headTop16(finBuf) {
    const { dev, dr, H, R, VOC } = this;
    const e = dev.createCommandEncoder(); e.copyBufferToBuffer(finBuf, H * 4, this.headIn.buffer, this.headIn.byteOffset ?? 0, R * H * 4); dev.queue.submit([e.finish()]); this.headS.compiled.collector.enqueue(this.headS.steps);
    const pass = dr.begin(); const t = dr.topk(pass, this.headLgBind, R, VOC); dr.end(pass);
    const [ov, oiF] = await Promise.all([dr.read(t.ov, R * 16), dr.read(t.oi, R * 16)]); const oi = new Uint32Array(oiF.buffer);
    const cand = [], unary = [];
    for (let r = 0; r < R; ++r) { cand.push(Array.from(oi.subarray(r * 16, r * 16 + 16))); unary.push(Array.from(ov.subarray(r * 16, r * 16 + 16))); }
    return { cand, unary };
  }

  async #draft(ctx, anchor) {
    const { dr, R } = this;
    const noise = this.#embed([anchor, ...Array(7).fill(CFG.MASK)]); const st = dr.draftStep(ctx, noise, 1.0);
    const hd = await this.#headTop16(st.final);
    const selH = await dr.read(st.selHidden, R * CFG.RANK); const sel = dr.select(anchor, hd.cand, hd.unary, selH);
    for (const l of st.layers) for (const b of Object.values(l)) b.destroy(); st.final.destroy(); st.selHidden.destroy(); st.h0.destroy();
    return sel.path;
  }

  // prompt lookup over context = hist + [anchor]: the most recent earlier position whose preceding tokens match the
  // context's suffix longest (>= ngramK, capped at 32); returns up to n tokens that followed it, or null
  #lookup(hist, anchor, n) {
    const L = hist.length, MAXM = 32; const at = (i) => i === L ? anchor : hist[i];
    let best = 0, bestP = -1;
    for (let p = L - 1; p >= 0; --p) {
      if (hist[p] !== anchor) continue;
      let m = 1; while (m < MAXM && p - m >= 0 && hist[p - m] === at(L - m)) m++;
      if (m > best) { best = m; bestP = p; if (m >= MAXM) break; }
    }
    if (best < this.ngramK) return null;
    const out = []; for (let i = bestP + 1; i <= L && out.length < n; ++i) out.push(at(i));
    return out;
  }

  // ---- the engine seam ----
  async *generate(tokenIds, cache, generationArgs, beginDecode, eosTokenId) {
    const { I, dr, rt, H } = this; const LV = this.block;
    if (tokenIds.length === 0) throw new Error('generation requires at least one input token');
    const { maxNewTokens, stopOnEos, onPrefillDone } = I.dc(generationArgs, {});
    const isEos = (t) => stopOnEos && I.fc(t, eosTokenId);
    const r = await this.#resources(cache); r.cache = cache;
    this.stats.generations++;
    // drafter context: keep it when it still mirrors the cache (prefix reuse without truncation), else rebuild from the suffix
    const past = cache.get_seq_length();
    if (r.ctxEnd !== past) dr.resetContext(r.ctx);
    // token history: truncate to the reused prefix; a cache filled outside this runner leaves it unknown (lookup only, so
    // a wrong history costs acceptance, never correctness)
    if (r.hist.length !== past) r.hist = r.hist.length > past ? r.hist.slice(0, past) : [];
    // prompt prefill in the engine's block sizes through tapped sessions; anchor = next token after the last chunk.
    // Only the last (sink + window) rows of the suffix feed the drafter context — earlier rows would be evicted anyway.
    const ids = Array.from(tokenIds); const keep = this.sink + this.window > 0 ? this.sink + this.window : ids.length; const firstKept = Math.max(0, ids.length - keep);
    let anchor = null;
    for (let i = 0; i < ids.length;) {
      const remaining = ids.length - i; const T = PREFILL_BLOCKS.find(b => b >= remaining) ?? PREFILL_BLOCKS[PREFILL_BLOCKS.length - 1];
      const chunk = ids.slice(i, i + T); const run = await this.#runBlock(r, await r.prefill(T), chunk); anchor = run.next;
      // the drafter's context projection takes <= 8 rows per call: walk the kept feature rows in 8-row slices
      for (let o = Math.max(0, firstKept - i); o < chunk.length; o += CFG.BLOCK) { const n = Math.min(CFG.BLOCK, chunk.length - o); dr.appendContext(r.ctx, { buffer: run.feat.buffer, offset: run.feat.offset + o * 5 * H * 4, size: n * 5 * H * 4 }, n); }
      i += chunk.length;
    }
    for (const t of ids) r.hist.push(t);
    await rt.queueIdle();
    onPrefillDone?.({ tokens: tokenIds.length, cache_length: cache.get_seq_length() });
    if (maxNewTokens <= 0 || isEos(anchor)) { r.ctxEnd = cache.get_seq_length(); return; }
    yield anchor; let emitted = 1;
    const eosSeen = { v: false };
    try {
      while (emitted < maxNewTokens && !eosSeen.v) {
        if (cache.get_seq_length() + LV > cache.maxLength) break;   // no room for a full verify block
        let drafted = this.ngram ? this.#lookup(r.hist, anchor, LV - 1) : null; const ng = !!drafted;
        if (!drafted || drafted.length < LV - 1) { const d = (await this.#draft(r.ctx, anchor)).slice(0, LV - 1); drafted = drafted ? [...drafted, ...d.slice(drafted.length)] : d; }
        const block = [anchor, ...drafted];
        r.slot.checkpoint.capture(); const pos = cache.seqLength;
        const run = await this.#runBlock(r, r.verify, block); const vt = Array.from(await run.tokens());
        let k = 0; while (k < drafted.length && vt[k] === drafted[k]) k++;
        const bonus = vt[k];
        // accepted drafts are already in the cache as rows 1..k; the drafter context takes rows 0..k of this run
        const out = [];
        for (let i = 0; i < k; ++i) { if (isEos(drafted[i])) { eosSeen.v = true; k = i; break; } out.push(drafted[i]); }
        if (!eosSeen.v) { if (isEos(bonus)) eosSeen.v = true; else out.push(bonus); }
        // rows of the verify block the cache keeps: anchor + accepted drafts (k+1), or fewer when maxNewTokens cuts
        // the output short (then the last yielded draft leaves the cache, like the engine's own loop)
        const y = Math.min(out.length, maxNewTokens - emitted);
        const keep = y < out.length ? y : k + 1;
        dr.appendContext(r.ctx, { ...run.feat, size: keep * 5 * H * 4 }, keep);
        if (keep < LV) { r.slot.checkpoint.restore(); r.rewind.run(keep); cache.seqLength = pos + keep; }
        for (let i = 0; i < keep; ++i) r.hist.push(block[i]);
        this.stats.cycles++; this.stats.accepted += k; if (ng) { this.stats.ngramCycles++; this.stats.ngramAccepted += k; }
        for (let i = 0; i < y; ++i) { yield out[i]; emitted++; this.stats.tokens++; }
        anchor = bonus;
      }
    } finally {
      await rt.queueIdle();
      r.ctxEnd = cache.get_seq_length();
    }
  }
}
