// Introspect the loaded engine: what the model object exposes, the lut2 pack layout, the device.
window.__gs = { state: 'init', log: [] };
(async () => {
  const V = window.__gs; const mark = (m) => { V.log.push(m); V.state = m; };
  try {
    const mod = await import('/engine.dflash.js?v=' + Date.now());
    const Eng = mod.TernaryBonsai2; mark('imported');
    const m = await Eng.load('/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf', { maxLength: 4096, onProgress: (ev) => { if (ev && ev.status) V.prog = `${ev.status} ${ev.loaded ?? ''}/${ev.total ?? ''}`; } });
    mark('loaded');
    const inner = m.model;
    const keys = (o) => { const out = []; for (let p = o; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) out.push(...Object.getOwnPropertyNames(p)); return [...new Set(out)]; };
    const describeT = (t) => t && ({ dtype: t.dtype, shape: t.shape, byteOffset: t.byteOffset, byteLength: t.byteLength, bufSize: t.buffer && t.buffer.size, keys: keys(t).slice(0, 30) });
    const r = {};
    r.mKeys = keys(m); r.innerKeys = keys(inner);
    r.runtimeKeys = keys(inner.runtime);
    const dev = inner.runtime.device;
    r.device = { features: [...dev.features], limits: Object.fromEntries(keys(dev.limits).filter(k => typeof dev.limits[k] === 'number').map(k => [k, dev.limits[k]])), adapterInfo: dev.adapterInfo && Object.fromEntries(['vendor','architecture','device','description','subgroupMinSize','subgroupMaxSize'].map(k => [k, dev.adapterInfo[k]])) };
    r.config = { hidden: inner.config.hidden_size, inter: inner.config.intermediate_size, vocab: inner.config.vocab_size, layers: inner.config.num_hidden_layers, layer_types: inner.config.layer_types.slice(0, 8), prismHadamard: inner.config.prismHadamard && { blockSize: inner.config.prismHadamard.blockSize, nWeights: inner.config.prismHadamard.weights.length, first: inner.config.prismHadamard.weights.slice(0, 6), inverse: inner.config.prismHadamard.inverseWeights, signWidths: Object.keys(inner.config.prismHadamard.signs) } };
    r.packs = Object.fromEntries(Object.entries(inner.packs).map(([k, v]) => [k, (v ?? []).map(p => ({ bits: describeT(p.bits), scales: describeT(p.scales), lut2: p.lut2, compact: p.compact, lut16: p.lut16, nIds: p.packedIds && p.packedIds.size, someIds: p.packedIds && [...p.packedIds].slice(0, 6), off0: p.offsets && (p.offsets.get ? p.offsets.get('layers.0.up_proj') : p.offsets['layers.0.up_proj']), lutId0: p.lutIds && (p.lutIds.get ? p.lutIds.get('layers.0.up_proj') : null), bc0: p.blockCounts && (p.blockCounts.get ? p.blockCounts.get('layers.0.up_proj') : p.blockCounts['layers.0.up_proj']) }))]));
    r.lmHead = { q4: describeT(inner.lmHeadQ4), q4s: describeT(inner.lmHeadQ4Scales), lut: inner.lmHeadLut, q8: describeT(inner.lmHeadQ8) };
    r.weights = describeT(inner.weights); r.offsetsTop = inner.offsets && inner.offsets.top; r.offsetsL0 = inner.offsets && inner.offsets.layers && inner.offsets.layers[0];
    r.weightsDtype = inner.weights && inner.weights.dtype;
    r.internals = Object.keys(Eng.__dflashInternals || {});
    V.results = r; mark('done');
  } catch (e) { V.error = String(e && e.stack || e); mark('error'); }
})();
