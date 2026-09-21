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
// (d) qwen35 verify mode: an optional 4th argument `$v` on lh(e,t,r) with {tapLayers, allRowsHead} semantics
// mirroring I0. No 4th argument -> the emitted graph is byte-identical to today's (every addition is gated on $v).
//   tapLayers: for each layer index Z in the list, StridedCopy the residual `Je` at the ENTRY of layer Z (I0's
//              convention: the output of layer Z-1) into scratch `dspark.features` [T, taps*H] (dtype = actGemm k,
//              as I0 uses `me = y`), declared as a graph output after the layer loop.
//   allRowsHead: per-row `ki(...)` argmax for every row into scratch `verify_tokens` [T] (I0's fallback head path,
//              the same call E0 makes for the decode head; ki dispatches internally to the two-stage Q4+Q8 rescore
//              head when both packs exist, else the single-stage LlamaDecodeLmHeadArgmax for the loaded pack), then
//              E0 takes next_token from verify_tokens[real_len-1] via nextTokenPick (K2.writeRunInputs already
//              writes `nextTokenPickUniform`).
const lhSig = 'function lh(e,t,r){';
once(new RegExp(lhSig.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 'lh signature');
out = out.replace(lhSig, 'function lh(e,t,r,$v){');
const lhEmbed = 'Je=Se(T0({g:J,w:re,weights:ke,model:e,ids:Oe,act:G,H:s,vocab:o,T:a,embedOffset:ie.embed_tokens}),"embed_tokens",!0);for(let te=0;te<i;++te){let ge=e.offsets.layers[te],';
once(new RegExp(lhEmbed.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 'lh embed + layer-loop head');
out = out.replace(lhEmbed,
  'Je=Se(T0({g:J,w:re,weights:ke,model:e,ids:Oe,act:G,H:s,vocab:o,T:a,embedOffset:ie.embed_tokens}),"embed_tokens",!0);' +
  'let $n=$v?.tapLayers?.length??0,$f=$n>0?J.scratch("dspark.features",k,[a,$n*s]):null;' +
  'for(let te=0;te<i;++te){if($f){let $i=$v.tapLayers.indexOf(te);$i>=0&&J.op("com.xenova.StridedCopy",{srcT:Je,dstT:$f},{args:{rows:a,srcStride:s,srcStart:0,dstStride:$n*s,dstStart:$i*s,copyCols:s}})}let ge=e.offsets.layers[te],');
const lhTail = 'Ee(te,ge,Je)}let Ze=J.nodeCount,st=E0({g:J,w:re,weights:ke,model:e,hidden:Je,act:G,H:s,vocab:o,T:a,eps:l,finalNormOffset:ie.model_norm,lmHeadOffset:ie.lm_head});return{graph:J.finish({name:"qwen35-prefill",params:{T:a}}),weights:me,states:ne,paramsName:"params",lastRowUniform:st,...L?{realRowsInput:L.name}:{},nextTokenTailNodeCount:J.nodeCount-Ze}}';
once(new RegExp(lhTail.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 'lh tail (E0 + return)');
for (const id of ['ki', 'E0']) once(new RegExp(`function ${id}\\(`), `internal function ${id}`);
out = out.replace(lhTail,
  'Ee(te,ge,Je)}$f&&J.output($f,"dspark.features");let Ze=J.nodeCount,$p;' +
  'if($v?.allRowsHead){let $t=J.scratch("verify_tokens","uint32",[a]);' +
  'for(let $r=0;$r<a;++$r)ki(J,{w:re,model:e,hidden:J.view(Je,$r*s,G,[s],`V.row${$r}`),weights:ke,ids:$t,hiddenSize:s,vocabSize:o,rmsEps:l,finalNormOffset:ie.model_norm,lmHeadOffset:ie.lm_head,q1:O,normedDtype:O&&S?"float16":"float32",nameSuffix:`V.r${$r}.`,outputOffset:$r});' +
  'J.output($t,"verify_tokens"),$p={tokens:$t,uni:J.uniform("next_token_pick_uni",32)}}' +
  'let st=E0({g:J,w:re,weights:ke,model:e,hidden:Je,act:G,H:s,vocab:o,T:a,eps:l,finalNormOffset:ie.model_norm,lmHeadOffset:ie.lm_head,...$p?{nextTokenPick:$p}:{}});' +
  'return{graph:J.finish({name:"qwen35-prefill",params:{T:a}}),weights:me,states:ne,paramsName:"params",lastRowUniform:st,...$p?{nextTokenPickUniform:$p.uni.name}:{},...L?{realRowsInput:L.name}:{},nextTokenTailNodeCount:J.nodeCount-Ze}}');
writeFileSync('engine.dflash.js', out);
console.log('wrote engine.dflash.js', out.length, 'bytes; features-output patch + internals hook + qwen35 verify-mode (lh 4th arg) applied');
