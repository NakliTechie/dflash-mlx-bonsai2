"""In-graph verify_block(8) timing through the runtime's own patch (install_prism_verify_linears) for each mode."""
import sys, os, time, statistics as st
sys.path.insert(0, 'lab')
PACK = '/Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit'
import mlx.core as mx
from prism_pack_loader import load_text_model
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.runtime.prism_qmm import install_prism_verify_linears
sys.path.insert(0, PACK + '/runtime'); import runtime as prism_rt
model, _ = load_text_model(PACK); ops = resolve_target_ops(model); ops.install_speculative_hooks(model)
CAP = {6, 20, 34, 48, 62}
def prefill():
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ids = mx.array([[248044] + [1000 + (i * 17) % 20000 for i in range(255)]])
    lg, _ = ops.forward_with_hidden_capture(model, input_ids=ids, cache=cache, capture_layer_ids=CAP); mx.eval(lg); return cache
cache = prefill(); vids = mx.array([[3000 + (i * 7) % 20000 for i in range(8)]])
def verify(N, reps=7):
    ts = []; last = None
    for r in range(reps):
        ops.arm_rollback(cache, prefix_len=256); ids = vids if N == 8 else mx.array([[3000]])
        t0 = time.perf_counter(); vl, _ = ops.verify_block(target_model=model, verify_ids=ids, target_cache=cache, capture_layer_ids=CAP); mx.eval(vl); ts.append((time.perf_counter() - t0) * 1000); last = vl
        ops.restore_after_acceptance(cache, target_len=256, acceptance_length=0, drafted_tokens=N)
    return st.median(ts[1:]), min(ts[1:]), last
for mode in sys.argv[1:] or ('v4b', 'v7', 'v4b', 'v7'):
    prism_rt.Packed._dflash_verify_mode = None      # force re-install
    install_prism_verify_linears(prism_rt.Packed, prism_rt.fwht, mode)
    m8, mn8, lg = verify(8); m1, mn1, _ = verify(1)
    print(f'{mode:5s}: verify(8) median {m8:6.1f} min {mn8:6.1f} ms | verify(1) {m1:5.1f} ms | ratio {m8/m1:.2f} | argmax {lg[0].argmax(-1).tolist()}')
