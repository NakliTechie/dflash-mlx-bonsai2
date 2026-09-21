# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 110.00 | n/a | 14.94 | n/a | n/a | n/a | 1201.51 ms | 12.75 GB | 0.62 | n/a | n/a | 92.52 | 92.52 |

- mode: smoke
- suite: smoke
- model: /Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
- draft: /Users/chiragpatnaik/Code/models/Qwen3.8-27B-DFlash2
- draft_quant: None
- git_hash: 789d067
- max_tokens: 2048
- block_tokens: 8
- repeat: 1
- cooldown: 10
- prompt_count: 1
- prompt_ids: smoke-custom-write-a-complete-production-quality-python-pack-48da619d
- prompt_source: smoke
- prompt_tokenization_mode: chat_template
- use_chat_template: True
- target_fa_window: 0
- draft_window: 64+1024
- verify_len_cap: 0
- verify_mode: dflash
- only_dflash: True

## Per Prompt

| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke-custom-write-a-complete-production-quality-python-pack-48da619d | 110 | n/a | 14.94 | n/a | n/a | n/a | 0.62 |
