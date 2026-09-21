# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 51.00 | n/a | 9.90 | n/a | n/a | n/a | 1443.03 ms | 12.75 GB | 0.60 | n/a | n/a | 35.82 | 35.82 |

- mode: smoke
- suite: smoke
- model: /Users/chiragpatnaik/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
- draft: /Users/chiragpatnaik/Code/models/Qwen3.8-27B-DFlash2-r3
- draft_quant: None
- git_hash: 6e7172b
- max_tokens: 2048
- block_tokens: 8
- repeat: 1
- cooldown: 10
- prompt_count: 1
- prompt_ids: smoke-custom-im-start-user-write-a-complete-short-story-of-d77c28af
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
| smoke-custom-im-start-user-write-a-complete-short-story-of-d77c28af | 51 | n/a | 9.90 | n/a | n/a | n/a | 0.60 |
