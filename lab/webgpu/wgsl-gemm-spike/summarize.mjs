// node summarize.mjs run.log -> compact table + run.json
import { readFileSync, writeFileSync } from 'node:fs';
const f = process.argv[2]; const t = readFileSync(f, 'utf8'); const i = t.indexOf('RESULTS');
console.log(t.slice(0, i).split('\n').filter(l => !/MODULE_TYPELESS|Reparsing|type.: .module|trace-warnings/.test(l)).join('\n'));
const J = JSON.parse(t.slice(i + 7)); const R = J.results; if (J.error) console.log('ERROR', J.error);
if (!R) process.exit();
writeFileSync(f.replace(/\.log$/, '.json'), JSON.stringify(R, null, 1));
const pick = (o) => o && (o.error ? { err: o.error } : { ms: o.gpuMedian, min: o.gpuMin, wall: o.wallMs ?? o.wallMsPerIter, ...(o.vsCpu ? { maxAbs: o.vsCpu.maxAbs, mean: o.vsCpu.meanAbs, argmax: o.vsCpu.argmaxMatch } : {}) });
const rows = [['decode gate/up (2 mats)', pick(R.engineDecodeGateUp)], ['decode gate/up after', R.engineDecodeGateUpAfter && { ms: R.engineDecodeGateUpAfter.gpuMedian }], ['decode gate/up tokens=4', pick(R.engineDecodeGateUpTokens4)], ['decode down (1 mat)', pick(R.engineDecodeDown)], ['prefill up M=8', pick(R.enginePrefillUp8)], ['lm_head engine M=1', pick(R.engineLmHead?.M1)], ['lm_head engine M=8', pick(R.engineLmHead?.M8)], ...(R.spike || []).map(s => [s.label || JSON.stringify(s.cfg), pick(s)]), ['lm_head spike ' + (R.lmHead?.label || ''), R.lmHead && (R.lmHead.error ? { err: R.lmHead.error } : { ms: R.lmHead.gpuMedian, min: R.lmHead.gpuMin, maxAbs: R.lmHead.vsCpuFirst2048Rows?.maxAbs, mean: R.lmHead.vsCpuFirst2048Rows?.meanAbs, argmax: R.lmHead.vsCpuFirst2048Rows?.argmaxMatch })]];
for (const [k, v] of rows) console.log(k.padEnd(44), JSON.stringify(v));
console.log('summary', JSON.stringify(R.summary));
