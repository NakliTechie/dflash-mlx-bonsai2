"""MLX oracle for the WebGPU drafter (stage-2 step 3). One speculation cycle on a fixed prompt, every stage dumped.

Extends lab/webgpu/dump_drafter_oracle.py: same prompt, same target pack (Ternary Bonsai 2, 2-bit prism pack),
round-3 DFlash2 drafter in bf16 (draft_quant=None -> the reference is exact). Stages (all fp32 .npy, batch dim
squeezed):
  context_features [C, 5H]      tapped residuals (layers 5/19/33/47/61 = capture ids 6/20/34/48/62) of the C prompt tokens
  draft_context    [C, H]       hidden_norm(fc(context_features))
  noise_embedding  [8, H]       target embed of [anchor, mask*7] (raw, BEFORE embed_scale; embed_scale in index.json)
  layer{i}_attn_in [8, H]       attention_conv.prepare(input_layernorm(h))  (the conv'd normed input to attention)
  layer{i}_attn_out[8, H]       h after the attention block (residual added)
  layer{i}_out     [8, H]       h after the MLP block (= layer output)
  final_hidden     [8, H]       norm(h)
  logits_top16     [7, 16]      the unary logits of the top-16 candidates per slot (rows 1..7 through the TARGET lm_head)
  cand_ids         [7, 16] i32  candidate ids (top-16 per slot, unordered as argpartition gives them)
  argmax           [7]    i32   logits argmax per slot
  head_rows        [7, 16, H]   effective TARGET lm_head rows for the candidates (lm_head is linear: Hadamard + ternary)
  sel_hidden       [7, 256]     hidden_projection(final_hidden[1:])
  sel_edges        [7, 16]      bilinear edge scores along the selected path
  selected         [7]    i32   the 7 drafted ids
  (the per-layer projected context cache is also dumped: layer{i}_ctx_k / ctx_v [C, 8, 128], k after k_norm + rope)
--weights gguf: replace the drafter's weights with the Q4_K_M GGUF's dequantized values (f16-rounded, exactly what the
browser holds) so the oracle isolates kernel error from quantization error. Writes to oracle/<tag>/.
"""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, 'lab'); sys.path.insert(0, os.path.expanduser('~/Code/llama.cpp-prism/gguf-py'))
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.runtime.prism_pack import load_pack_tokenizer
from dflash_mlx.runtime.loading import load_draft_bundle
from dflash_mlx.draft_backend import EagerDraftBackend

ap = argparse.ArgumentParser(); ap.add_argument('--weights', choices=['bf16', 'gguf'], default='bf16'); ap.add_argument('--out', default=None)
ap.add_argument('--features', default='lab/webgpu/oracle/context_features.npy', help='cached tapped features [C, 5H] from a full-target prefill (dump_drafter_oracle.py); if missing, the full target is loaded and run')
ap.add_argument('--meta', default='lab/webgpu/oracle/meta.json', help='meta.json next to the cached features (prompt_ids, anchor)')
args = ap.parse_args()
PACK = os.path.expanduser('~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit')
DRAFT = os.path.expanduser('~/Code/models/Qwen3.8-27B-DFlash2-r3')
GGUF = os.path.join(DRAFT, 'Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf')
OUT = args.out or f'lab/webgpu/drafter/oracle/{args.weights}'; os.makedirs(OUT, exist_ok=True)
H = 5120; BLOCK = 8; TOPK = 16
prompt = '<|im_start|>user\nWrite a Python function that reverses a string, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
index = {'prompt': prompt, 'weights': args.weights, 'stages': {}}
def save(name, arr):
    a = np.array(arr.astype(mx.float32) if mx.issubdtype(arr.dtype, mx.floating) else arr.astype(mx.int32)) if isinstance(arr, mx.array) else np.asarray(arr)
    np.save(f'{OUT}/{name}.npy', a); index['stages'][name] = {'shape': list(a.shape), 'dtype': str(a.dtype)}; print(f'  {name} {a.shape} {a.dtype}')
tok = load_pack_tokenizer(PACK); enc = lambda s: tok.encode(s, add_special_tokens=False) if hasattr(tok, 'encode') else tok(s)['input_ids']
ids = list(enc(prompt)); CAP = {6, 20, 34, 48, 62}; C = len(ids)

# ---- target side. The drafter borrows two target modules (embed_tokens, lm_head); the tapped features come from a full
# prefill. Memory: the full pack is 8 GB, the two modules ~0.7 GB, so the prefill is cached and the light path is default.
class _Ops:   # the two target_ops entry points the drafter uses, on the pack's own Packed modules (runtime/runtime.py)
    def __init__(self, embed, head): self._embed, self._head = embed, head
    def embed_tokens(self, _m): return self._embed
    def logits_from_hidden(self, _m, h): return self._head(h)
if os.path.exists(args.features) and os.path.exists(args.meta):
    meta = json.load(open(args.meta)); assert meta['prompt_ids'] == ids, 'cached features are for a different prompt'
    feats = mx.array(np.load(args.features)).astype(mx.float32); anchor = int(meta['anchor']); print('cached features', feats.shape, 'anchor', anchor, repr(tok.decode([anchor])), 'from', args.features)
    sys.path.insert(0, os.path.join(PACK, 'runtime')); from runtime import Packed
    cfg = json.load(open(os.path.join(PACK, 'config.json'))); W = mx.load(os.path.join(PACK, 'model.safetensors'))   # lazy: only the touched arrays materialize
    def packed(path):
        rec = next(r for r in cfg['modules'] if r['path'] == path); key = 'language_model.' + path
        return Packed([W[key + '.weight'], W[key + '.scales'], W[key + '.biases']], rec['block'], W.get(key + '.signs'), rec['embedding'], mx.float16)
    ops = _Ops(packed('model.embed_tokens'), packed('lm_head')); model = None
    class _TM: embed_scale = 1.0
    class _TOps:                              # bind_target_model only reads text_model(...).embed_scale (qwen3.5 has none -> 1.0)
        def text_model(self, _m): return _TM()
    bind_ops = _TOps()
else:
    model, _ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model); bind_ops = ops
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    logits, captured = ops.forward_with_hidden_capture(model, input_ids=mx.array([ids]), cache=cache, capture_layer_ids=CAP); mx.eval(logits)
    anchor = int(logits[0, -1].argmax()); print('prompt tokens', C, 'anchor', anchor, repr(tok.decode([anchor])))
    feats = mx.concatenate([captured[k][0] for k in sorted(captured)], axis=-1).astype(mx.float32); mx.eval(feats)
    del logits, captured, cache
save('context_features', feats)

# ---- drafter ----
bundle = load_draft_bundle(DRAFT, draft_quant=None)
draft = bundle.model if hasattr(bundle, 'model') else bundle[0]
draft.bind_target_model(model, target_ops=bind_ops)
print('draft', type(draft).__name__, 'embed_scale', draft.embed_scale, 'mask', draft.mask_token_id, 'block', draft.block_size)
if args.weights == 'gguf':
    import gguf
    from gguf.quants import dequantize
    reader = gguf.GGUFReader(GGUF)
    def mlx_name(g):
        if g == 'fc.weight': return 'fc.weight'
        if g == 'enc.output_norm.weight': return 'hidden_norm.weight'
        if g == 'output_norm.weight': return 'norm.weight'
        if g == 'selector_hidden.weight': return 'candidate_selector.hidden_projection.weight'
        if g == 'selector_predecessor.weight': return 'candidate_selector.predecessor_codebook.weight'
        if g == 'selector_successor.weight': return 'candidate_selector.successor_codebook.weight'
        p = g.split('.'); n = p[1]; rest = '.'.join(p[2:])
        m = {'attn_q.weight': 'self_attn.q_proj.weight', 'attn_k.weight': 'self_attn.k_proj.weight', 'attn_v.weight': 'self_attn.v_proj.weight', 'attn_output.weight': 'self_attn.o_proj.weight',
             'attn_q_norm.weight': 'self_attn.q_norm.weight', 'attn_k_norm.weight': 'self_attn.k_norm.weight', 'attn_norm.weight': 'input_layernorm.weight', 'ffn_norm.weight': 'post_attention_layernorm.weight',
             'ffn_gate.weight': 'mlp.gate_proj.weight', 'ffn_up.weight': 'mlp.up_proj.weight', 'ffn_down.weight': 'mlp.down_proj.weight',
             'attn_conv_base': 'attention_conv.base_kernel', 'attn_conv_proj.weight': 'attention_conv.kernel_projection.weight',
             'ffn_conv_base': 'mlp_conv.base_kernel', 'ffn_conv_proj.weight': 'mlp_conv.kernel_projection.weight'}
        return f'layers.{n}.{m[rest]}'
    new = []
    for t in reader.tensors:
        shape = [int(x) for x in t.shape][::-1]
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            arr = np.array(t.data, dtype=np.float32).reshape(shape); dt = mx.float32
        else:
            arr = dequantize(np.array(t.data), t.tensor_type).reshape(shape).astype(np.float16); dt = mx.float16   # f16-rounded = the browser's copy
        new.append((mlx_name(t.name), mx.array(arr).astype(dt)))
    draft.load_weights(new, strict=True); mx.eval(draft.parameters()); print('loaded', len(new), 'tensors from the GGUF (quantized -> f16, norms/bases f32)')
    index['gguf'] = GGUF
index.update({'prompt_ids': ids, 'C': C, 'H': H, 'anchor': anchor, 'anchor_text': tok.decode([anchor]), 'mask_token_id': draft.mask_token_id, 'block_size': BLOCK,
              'embed_scale': float(draft.embed_scale), 'taps': [5, 19, 33, 47, 61], 'capture_layer_ids': sorted(CAP), 'sliding_window': draft.layers[0].self_attn.sliding_window,
              'rope_theta': float(draft.args.rope_theta), 'head_dim': 128, 'n_heads': 32, 'n_kv_heads': 8, 'rms_eps': float(draft.args.rms_norm_eps),
              'output_multiplier': float(draft.args.output_multiplier), 'final_logit_softcapping': draft.args.final_logit_softcapping})

# ---- stage 1: projected context ----
draft_context = draft.project_target_hidden(feats[None]); mx.eval(draft_context); save('draft_context', draft_context[0])
# ---- stage 2: noise embedding ----
block_ids = mx.array([anchor] + [draft.mask_token_id] * (BLOCK - 1))
noise = ops.embed_tokens(model)(block_ids[None]); mx.eval(noise); save('noise_embedding', noise[0])
print('noise dtype', noise.dtype, 'features dtype', feats.dtype)
# ---- stage 3: layers, one at a time with the same caches propose_block would use ----
backend = EagerDraftBackend(); caches = backend.make_cache(draft_model=draft, sink_size=64, window_size=1024)
h = noise * draft.embed_scale
for i, (layer, c) in enumerate(zip(draft.layers, caches)):
    res = h
    x, dyn = layer.attention_conv.prepare(layer.input_layernorm(h)); mx.eval(x); save(f'layer{i}_attn_in', x[0])
    a = layer.self_attn(x, target_hidden=draft_context, cache=c)
    h = res + layer.attention_conv.finish(a, dyn); mx.eval(h); save(f'layer{i}_attn_out', h[0])
    k, v = c.fetch(); save(f'layer{i}_ctx_k', k[0].transpose(1, 0, 2)); save(f'layer{i}_ctx_v', v[0].transpose(1, 0, 2))   # [C, kv_heads, 128]
    res = h
    x, dyn = layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
    h = res + layer.mlp_conv.finish(layer.mlp(x), dyn); mx.eval(h); save(f'layer{i}_out', h[0])
final = draft.norm(h); mx.eval(final); save('final_hidden', final[0])
# cross-check: the whole-model call on fresh caches gives the same final hidden
caches2 = backend.make_cache(draft_model=draft, sink_size=64, window_size=1024)
final2 = draft.forward_projected_context(noise_embedding=noise, draft_context=draft_context, cache=caches2); mx.eval(final2)
index['selfcheck_forward_max_abs'] = float(mx.abs(final2 - final).max()); print('self-check forward_projected_context max|diff|', index['selfcheck_forward_max_abs'])
# ---- stage 4: target lm_head on rows 1..7, candidates, selector ----
rows = final[:, 1:, :]
logits = draft.compute_logits(ops.logits_from_hidden(model, rows)); mx.eval(logits)
print('logits', logits.shape, logits.dtype)
cand = mx.argpartition(logits, -TOPK, axis=-1)[..., -TOPK:]; unary = mx.take_along_axis(logits, cand, axis=-1); mx.eval(cand, unary)
save('cand_ids', cand[0]); save('logits_top16', unary[0]); save('argmax', logits[0].argmax(axis=-1))
# effective head rows: lm_head is linear in its input, so feed the identity through it in chunks and gather the candidate columns
cand_np = np.array(cand[0]); rows_eff = np.zeros((BLOCK - 1, TOPK, H), np.float32)
for s in range(0, H, 1024):
    eye = mx.zeros((1, 1024, H), dtype=mx.float32); eye[0, :, s:s + 1024] = mx.eye(1024)
    cols = draft.compute_logits(ops.logits_from_hidden(model, eye))[0].astype(mx.float32); mx.eval(cols)   # [1024, V]
    cols_np = np.array(cols[:, mx.array(cand_np.reshape(-1))])                                             # [1024, 7*16]
    rows_eff[:, :, s:s + 1024] = cols_np.T.reshape(BLOCK - 1, TOPK, 1024)
save('head_rows', rows_eff)
recon = np.einsum('sch,sh->sc', rows_eff, np.array(rows[0].astype(mx.float32)))
index['head_rows_reconstruction_max_abs'] = float(np.abs(recon - np.array(unary[0].astype(mx.float32))).max()); print('head_rows reconstruction max|diff| vs logits', index['head_rows_reconstruction_max_abs'])
sel = draft.candidate_selector
hp = sel.hidden_projection(rows); mx.eval(hp); save('sel_hidden', hp[0])
succ = sel.successor_codebook(cand)
pred = mx.array([anchor]); path = []; edges_all = []
for pos in range(BLOCK - 1):
    edges = sel._edge_scores(pred[:, None], succ[:, pos], hp[:, pos])[:, 0]
    scores = unary[:, pos] + edges; chosen = mx.argmax(scores, axis=-1)
    pred = mx.take_along_axis(cand[:, pos], chosen[:, None], axis=-1)[:, 0]; mx.eval(pred, edges)
    path.append(int(pred[0])); edges_all.append(np.array(edges[0].astype(mx.float32)))
save('sel_edges', np.stack(edges_all)); save('selected', np.array(path, np.int32))
proposal = draft.select_proposal(draft_hidden=rows, logits=ops.logits_from_hidden(model, rows), anchor_ids=mx.array([anchor]), temperature=0.0)
official = [int(x) for x in np.array(proposal.token_ids)]
index['selected'] = path; index['selected_official'] = official; index['selected_text'] = tok.decode(path); index['argmax_text'] = tok.decode([int(x) for x in np.array(logits[0].argmax(axis=-1))])
print('selected', path, repr(index['selected_text']), 'official select_proposal', official, 'match', path == official)
json.dump(index, open(f'{OUT}/index.json', 'w'), indent=1); print('wrote', OUT)
