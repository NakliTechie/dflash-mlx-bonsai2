# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 43.00 | n/a | 19.04 | n/a | n/a | n/a | 1412.53 ms | 12.52 GB | 0.77 | n/a | n/a | 30.90 | 30.90 |

- mode: smoke
- suite: smoke
- model: /Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
- draft: /Users/chiragpatnaik/Code/models/Qwen3.8-27B-DFlash2
- draft_quant: None
- git_hash: b268b88
- max_tokens: 512
- block_tokens: 8
- repeat: 1
- cooldown: 10
- prompt_count: 1
- prompt_ids: smoke-custom-def-lru-cache-class-return-a-class-lru-a29b3ef1
- prompt_source: smoke
- prompt_tokenization_mode: raw
- use_chat_template: False
- target_fa_window: 0
- draft_window: 64+1024
- verify_len_cap: 0
- verify_mode: dflash
- only_dflash: True

## Per Prompt

| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke-custom-def-lru-cache-class-return-a-class-lru-a29b3ef1 | 43 | n/a | 19.04 | n/a | n/a | n/a | 0.77 |
