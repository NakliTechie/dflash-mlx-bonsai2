// Node self-test: parse the drafter GGUF and compare dequantized tensors against gguf-py (numpy) on a few tensors.
// Usage: node test-gguf.mjs <gguf> ; writes /tmp-free scratch under oracle/.. no: reads python output from stdin pipe.
import fs from 'node:fs'; import { execFileSync } from 'node:child_process';
import { readGGUF, dequantize, fsSource, typeName } from './gguf.js';
const path = process.argv[2]; const fd = fs.openSync(path, 'r');
const g = await readGGUF(fsSource(fd, fs));
console.log('version', g.version, 'tensors', g.tensors.length, 'dataStart', g.dataStart, 'arch', g.kv['general.architecture'], 'mask', g.kv['tokenizer.ggml.mask_token_id']);
const names = ['blk.0.attn_k_norm.weight', 'blk.0.attn_conv_base', 'blk.0.attn_k.weight', 'blk.2.attn_v.weight', 'selector_hidden.weight', 'blk.2.ffn_down.weight'];
const py = `
import sys, os, json, numpy as np
sys.path.insert(0, os.path.expanduser('~/Code/llama.cpp-prism/gguf-py'))
import gguf
from gguf.quants import dequantize
r = gguf.GGUFReader(sys.argv[1]); out = {}
for t in r.tensors:
    if t.name in ${JSON.stringify(names)}:
        a = np.array(t.data, dtype=np.float32) if t.tensor_type == gguf.GGMLQuantizationType.F32 else dequantize(np.array(t.data), t.tensor_type)
        a = a.reshape(-1).astype(np.float32); out[t.name] = [float(a[i]) for i in [min(i, len(a)-1) for i in [0,1,2,3,255,256,1000, len(a)//2, len(a)-1]]] + [float(np.abs(a).sum())]
print(json.dumps(out))`;
const ref = JSON.parse(execFileSync('/Users/chiragpatnaik/Code/dflash-mlx-bonsai2/.venv/bin/python', ['-c', py, path]).toString());
let bad = 0;
for (const name of names) {
  const t = g.byName[name]; const raw = await fsSource(fd, fs)(t.absOffset, t.bytes); const t0 = Date.now(); const a = dequantize(t.type, raw, t.n); const ms = Date.now() - t0;
  const idx = [0, 1, 2, 3, 255, 256, 1000, (t.n / 2) | 0, t.n - 1].map(i => Math.min(i, t.n - 1)); let sum = 0; for (let i = 0; i < a.length; ++i) sum += Math.abs(a[i]);
  const mine = idx.map(i => a[i]).concat([sum]); const r = ref[name];
  const maxRel = Math.max(...mine.map((v, i) => Math.abs(v - r[i]) / Math.max(1e-6, Math.abs(r[i]))));
  const ok = maxRel < 1e-4; if (!ok) bad++;
  console.log(ok ? 'OK ' : 'BAD', name, typeName(t.type), t.ne, `${ms} ms`, 'maxRel', maxRel.toExponential(2), 'sample', mine.slice(0, 4).map(v => v.toFixed(5)), 'ref', r.slice(0, 4).map(v => v.toFixed(5)));
}
console.log(bad ? `FAIL ${bad}` : 'ALL OK');
process.exit(bad ? 1 : 0);
