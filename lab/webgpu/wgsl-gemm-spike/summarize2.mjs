import { readFileSync, writeFileSync } from 'node:fs';
const f = process.argv[2]; const t = readFileSync(f, 'utf8'); const i = t.indexOf('RESULTS');
console.log(t.slice(0, i).split('\n').filter(l => !/MODULE_TYPELESS|Reparsing|type.: .module|trace-warnings/.test(l)).join('\n'));
const J = JSON.parse(t.slice(i + 7)); const R = J.results; if (J.error) console.log('ERROR', J.error); if (!R) process.exit();
writeFileSync(f.replace(/\.log$/, '.json'), JSON.stringify(R, null, 1));
console.log('engine', JSON.stringify(R.engine), 'prefill', JSON.stringify(R.enginePrefill));
for (const [k, v] of Object.entries(R.runs || {})) { console.log(`== ${k} decode ${v.decodeMs} ms (ref rows ${v.refRows}, cpu ${v.refMs} ms) best: ${v.best} ratio ${v.bestRatio}`); for (const x of v.variants) console.log('  ', (x.label || '').padEnd(46), x.error ? 'ERR ' + x.error.slice(0, 100) : `${x.gpuMedian} ms (min ${x.gpuMin}) ratio ${x.ratio}  maxAbs ${x.vsCpu.maxAbs} mean ${x.vsCpu.meanAbs} argmax ${x.vsCpu.argmaxMatch}`); }
