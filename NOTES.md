# DFlash-MLX Notes

## TODO: Gemma 4 DFlash Draft Model

We want to create DFlash draft models for Gemma 4 (specifically `mlx-community/gemma-4-26b-a4b-it-4bit` and `mlx-community/gemma-4-31b-it-4bit`).

### Blockers
- z-lab training code not yet public (promised soon)
- Inference code has ~5 Qwen-specific coupling points that need adaptation

### When training code drops
1. Adapt inference code in `model.py` and `runtime.py` for Gemma 4 architecture
2. Run training: ~800K samples, cache target hidden states, train 5-layer draft model
3. Publish draft model to HuggingFace

### References
- Paper: https://arxiv.org/abs/2602.06036
- z-lab DFlash collection: https://huggingface.co/collections/z-lab/dflash
- z-lab GitHub (watch for training code release): https://github.com/z-lab/dflash
