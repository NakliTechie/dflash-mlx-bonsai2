"""Text-only loader for PrismML Hadamard MLX packs (schema 2, VL packs included).

Builds the stock mlx_lm Qwen3.5 TextModel and installs the pack's Packed modules into it,
skipping the vision tower. Mirrors runtime/artifact.load_model but accepts the
`language_model.` tensor namespace of vision packs.
"""
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from mlx_lm.utils import load_tokenizer

def load_text_model(directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    if config.get('model_type') != 'prism_hadamard_qwen35' or config.get('base_model_type') != 'qwen3_5':
        raise ValueError('Unsupported packed model schema')
    sys.path.insert(0, str(directory / 'runtime'))
    from runtime import Packed
    prefix = 'language_model.' if config.get('components', {}).get('vision') else ''
    model = TextModel(TextModelArgs.from_dict(config['text_config']))
    weights = mx.load(str(directory / 'model.safetensors'))
    seen = set()
    for record in config['modules']:
        path = record['path']
        if path in seen: raise ValueError('Duplicate packed module')
        seen.add(path)
        parts = path.split('.'); parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = prefix + path
        arrays = [weights[key + '.' + s] for s in ('weight', 'scales', 'biases')]
        if record['dtype'] != 'float16': raise ValueError('Unsupported activation dtype')
        block = record['block']
        if block and block not in (512, 1024, 2048, 4096): raise ValueError('Unsupported block size')
        signs = weights.get(key + '.signs')
        if block and signs is None: raise ValueError('Missing sign vector')
        setattr(parent, parts[-1], Packed(arrays, block, signs, record['embedding'], mx.float16))
    text_weights = [(k[len(prefix):], v) for k, v in weights.items() if k.startswith(prefix) and not k.startswith('vision_tower.')]
    model.load_weights(text_weights, strict=True)
    model.eval(); mx.eval(model.parameters())
    from runtime import fwht
    from dflash_mlx.runtime.prism_qmm import install_prism_verify_linears
    config = dict(config)
    config['dflash_prism_verify'] = install_prism_verify_linears(Packed, fwht)
    return model, config

def load_pack_tokenizer(directory):
    return load_tokenizer(Path(directory))
