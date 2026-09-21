"""Write the round-3 DFlash2 drafter (HF-layout safetensors) as a `dflash`-arch GGUF using z-lab's official GGUF
as the metadata template (all KV fields incl. tokenizer copied; tensor names mapped 1:1; norms/conv bases F32,
matrices BF16). Quantize afterwards with llama-quantize to Q4_K_M. Usage: r3_to_gguf.py REF.gguf R3_DIR OUT.gguf"""
import sys, os, json, struct, numpy as np, torch
sys.path.insert(0, os.path.expanduser('~/Code/llama.cpp-prism/gguf-py'))
import gguf
from safetensors import safe_open
ref, r3dir, out = sys.argv[1:4]
reader = gguf.GGUFReader(ref)
writer = gguf.GGUFWriter(out, 'dflash')
skip = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count', 'general.file_type', 'general.quantization_version', 'general.architecture'}
n_kv = 0
for name, field in reader.fields.items():
    if name in skip: continue
    t0 = field.types[0]
    if t0 == gguf.GGUFValueType.ARRAY:
        sub = field.types[1]
        if sub == gguf.GGUFValueType.STRING: vals = [bytes(field.parts[i]).decode('utf-8') for i in field.data]
        else: vals = [field.parts[i].tolist()[0] for i in field.data]
        writer.add_array(name, vals)
    elif t0 == gguf.GGUFValueType.STRING:
        v = bytes(field.parts[field.data[0]]).decode('utf-8')
        if name == 'general.name': v = 'Qwen3.8-27B-DFlash2-ternary-bonsai2-r3'
        if name == 'general.finetune': v = 'DFlash2 re-fit on Ternary-Bonsai-2-27B generations (round 3)'
        writer.add_string(name, v)
    else:
        v = field.parts[field.data[0]].tolist()[0]
        {gguf.GGUFValueType.UINT32: writer.add_uint32, gguf.GGUFValueType.INT32: writer.add_int32, gguf.GGUFValueType.FLOAT32: writer.add_float32,
         gguf.GGUFValueType.BOOL: writer.add_bool, gguf.GGUFValueType.UINT64: writer.add_uint64, gguf.GGUFValueType.UINT8: writer.add_uint8}[t0](name, v)
    n_kv += 1
print('copied', n_kv, 'KV fields')
def gname(k):
    if k == 'fc.weight': return 'fc.weight'
    if k == 'hidden_norm.weight': return 'enc.output_norm.weight'
    if k == 'norm.weight': return 'output_norm.weight'
    if k == 'candidate_selector.hidden_projection.weight': return 'selector_hidden.weight'
    if k == 'candidate_selector.predecessor_codebook.weight': return 'selector_predecessor.weight'
    if k == 'candidate_selector.successor_codebook.weight': return 'selector_successor.weight'
    p = k.split('.'); assert p[0] == 'layers'; n = p[1]; rest = '.'.join(p[2:])
    m = {'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
         'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight', 'input_layernorm.weight': 'attn_norm.weight', 'post_attention_layernorm.weight': 'ffn_norm.weight',
         'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.down_proj.weight': 'ffn_down.weight',
         'attention_conv.base_kernel': 'attn_conv_base', 'attention_conv.kernel_projection.weight': 'attn_conv_proj.weight',
         'mlp_conv.base_kernel': 'ffn_conv_base', 'mlp_conv.kernel_projection.weight': 'ffn_conv_proj.weight'}
    return f'blk.{n}.{m[rest]}'
refshape = {t.name: [int(x) for x in t.shape] for t in reader.tensors}
reftype = {t.name: t.tensor_type for t in reader.tensors}
written = set()
with safe_open(os.path.join(r3dir, 'model.safetensors'), framework='pt') as f:
    for k in sorted(f.keys()):
        g = gname(k); t = f.get_tensor(k)
        assert list(t.shape)[::-1] == refshape[g], (k, g, list(t.shape), refshape[g])
        if reftype[g] == gguf.GGMLQuantizationType.F32:
            writer.add_tensor(g, t.float().numpy())
        else:
            writer.add_tensor(g, t.contiguous().view(torch.int16).numpy().view(np.uint16), raw_dtype=gguf.GGMLQuantizationType.BF16)
        written.add(g)
missing = set(refshape) - written; assert not missing, missing
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(progress=False); writer.close()
print('wrote', out, os.path.getsize(out) / 1e9, 'GB,', len(written), 'tensors')
