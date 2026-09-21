// Stage-2 step 1 (scratchpad only, not the repo): expose the vendored engine's internal graph-session
// pieces so a harness can build the verify graph (I0 with the dspark config) and read `verify_tokens` +
// `dspark.features` back. Marker-guarded like scripts/extract-ternary-bonsai-2-27b.mjs.
import { readFileSync, writeFileSync } from 'node:fs';
const src = readFileSync(process.argv[2] ?? '/Users/chiragpatnaik/Code/naklios-universe/LocalMind/ternary_bonsai_2_27b.js', 'utf8');
let out = src;
const once = (re, what) => { const m = out.match(new RegExp(re.source, re.flags + 'g')); if (!m || m.length !== 1) throw new Error(`${what}: expected 1 match, got ${m ? m.length : 0}`); };
// (a) declare dspark.features as a graph output so compiled.tensor("dspark.features") is readable.
// Declared at the point where the all-rows head already declares verify_tokens (an expression sequence, so an
// extra comma-expression is legal there; the `let` declarator list where the scratch is created is not).
const featRe = /B\.output\(L,"verify_tokens"\),we=\{tokens:L,/;
once(featRe, 'verify_tokens output site');
out = out.replace(featRe, 'B.output(L,"verify_tokens"),ne&&B.output(ne,"dspark.features"),we={tokens:L,');
// (b) internals on the exported class, inserted before the export line (same module scope).
const exportLine = 'export{Cl as DEFAULT_GGUF_FILE,Ri as DEFAULT_MODEL_ID,zl as TernaryBonsai2,';
once(new RegExp(exportLine.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 'export line');
// I0/X2 are hoisted functions; K2/M0/f0 are assigned inside esbuild lazy-init wrappers (A(() => {...})) that run on
// first use, so expose them through getters that resolve at access time (after the model has loaded).
for (const id of ['I0', 'X2']) once(new RegExp(`function ${id}\\(`), `internal function ${id}`);
for (const id of ['M0', 'K2', 'f0', 'ch']) once(new RegExp(`\\b${id}=class`), `internal class ${id}`);
once(/function lh\(e,t,r\)\{/, 'internal function lh (qwen35 prefill emission)');
once(/var ch,dh=A\(/, 'lazy init wrapper dh for ch');
// (c) wgsl-gemm-spike additions: the graph builder (ba, lazily assigned by wi()), the weight/pack binders and
// the compile entry point, so a harness can emit a one-op micro graph around the engine's own decode / prefill
// matmul ops and time them in isolation.
once(/var u0,ba,wi=A\(/, 'lazy init wrapper wi for ba');
for (const id of ['_i', 'fi', 'bi', 's0', 'Xs', 'et', 'Cu', 'pi', 'a0', 'N2']) once(new RegExp(`function ${id}\\(`), `internal function ${id}`);
out = out.replace(exportLine, 'zl.__dflashInternals={get I0(){return I0},get X2(){return X2},get M0(){return M0},get K2(){return K2},get f0(){return f0},get ch(){dh();return ch},get lh(){return lh},get ba(){wi();return ba},get _i(){return _i},get fi(){return fi},get bi(){return bi},get s0(){return s0},get Xs(){return Xs},get et(){return et},get Cu(){return Cu},get pi(){return pi},get a0(){return a0},get N2(){return N2}};' + exportLine);
writeFileSync('engine.dflash.js', out);
console.log('wrote engine.dflash.js', out.length, 'bytes; features-output patch + internals hook applied');
