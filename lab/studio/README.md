# Running the Bonsai 2 DFlash work on Om's Mac Studio (M4 Max · 64 GB)

Box + access + the cooperative lock: `~/Code/infra/remote-studio/README.md` (Tailscale IP `100.89.200.65`,
user `ompatnaik`, key-based ssh; take `studio-lock.sh` before any heavy job). Both Tailscales must be UP —
on 2026-09-21 23:10 the laptop's was STOPPED (`/Applications/Tailscale.app/Contents/MacOS/Tailscale up`) and the
Studio's is off until Om turns it on.

Why the Studio for this project: 64 GB (a 100k-token MLX prefill needs ~21 GB, the M4 Pro has 24 total),
2× the GPU bandwidth (plain decode ~40 tok/s expected), and a machine nobody else is using while it runs.
Numbers measured there are a DIFFERENT chip — label them M4 Max in any table; the README's M4 Pro numbers stay.

## One-time sync (laptop → Studio; the Studio's own HF downloads are ~1.6 MB/s, do not download there)

```bash
bash lab/studio/sync.sh              # repo (no .git/.venv/lab logs) + pack + round-3 drafter + PTQ1_0 GGUF, rsync over Tailscale
bash lab/studio/remote.sh setup      # uv venv, pip install -e ., dflash doctor
```

## Jobs (each wrapped in the lock; results rsync back into lab/studio/results/)

```bash
bash lab/studio/remote.sh bench      # leg-7 style A/B: plain decode vs DFlash (v7, round-3), thinking off, 4 prompts x2
bash lab/studio/remote.sh prefill100k   # MiaAI-Lab cold-prefill protocol via dflash serve: 8k, 32k, 100k
bash lab/studio/remote.sh mtp        # (after the MTP GGUF is on the box) llama.cpp fork draft-mtp vs plain
```
