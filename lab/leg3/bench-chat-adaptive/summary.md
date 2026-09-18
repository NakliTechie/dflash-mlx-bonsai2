# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 80.00 | n/a | 19.04 | n/a | n/a | n/a | 941.11 ms | 12.51 GB | 0.58 | n/a | n/a | 86.30 | 86.30 |

- mode: smoke
- suite: smoke
- model: /Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
- draft: /Users/chiragpatnaik/Code/models/Qwen3.8-27B-DFlash2
- draft_quant: None
- git_hash: 4460d3b
- max_tokens: 512
- block_tokens: 8
- repeat: 1
- cooldown: 10
- prompt_count: 1
- prompt_ids: smoke-custom-write-a-warm-two-paragraph-email-to-a-friend-de-2050ddb2
- prompt_source: smoke
- prompt_tokenization_mode: chat_template
- use_chat_template: True
- target_fa_window: 0
- draft_window: 64+1024
- verify_len_cap: 0
- verify_mode: adaptive
- only_dflash: True

## Per Prompt

| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke-custom-write-a-warm-two-paragraph-email-to-a-friend-de-2050ddb2 | 80 | n/a | 19.04 | n/a | n/a | n/a | 0.58 |
