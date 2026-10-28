# Contributing

## Never commit

`.gitignore` covers these; please don't force-add around it.

- **Credentials and keys** (`*.pem`, `.env`, tokens).
- **Datasets, audio and dataset-derived transcripts** (`ASR/data/`, `*.wav`, `ASR/results/**/*.jsonl`).
  KsponSpeech in particular is **not cleared for redistribution**.
- **Model weights and checkpoints** (`*.gguf`, `*.safetensors`, `*.onnx`, `ASR/weights/`, `ASR/runs/`).
- **Phone packages, firmware images, build trees, vendor binaries** (`s2st-qdc/`, `firmware/`, `build*/`, `third_party/`).
- **Local tool state** (`.claude/`, `.opencode/`).

Do commit: code, the recipe config, docs. **Keep testing code, probes, run logs and raw results out of `ASR/` and
`harness/`**: put the result and the verdict in that folder's README (and the full story in the WORKLOG).

## How we record work

- **One experiment = one config + one hypothesis + one gate, written down before running.** The ASR configs carry
  `_hypothesis` / `_gate` keys for this.
- **Results go into `docs/history/WORKLOG.md`** as a new section: what was tested, the numbers (paired, with n), what
  it means, and what went wrong. Negative results are kept; several of the most useful findings are in them.
  Update the results/verdict tables in `ASR/README.md` or `harness/README.md` when the headline changes.
- **Compare on identical data** and report paired deltas with a bootstrap CI (`ASR/scripts/bootstrap_ci.py`).
- **Measure the shipped artifact.** PyTorch numbers are not GGUF numbers, and host numbers are not phone numbers.

## Lessons that cost real time (read before running anything long)

| trap | do instead |
|---|---|
| `pgrep -f` / `pkill -f <script>` matches its own shell; a dead run looked alive for 3.4 h, and a kill took out the caller | match the binary name (`pidof`) or a bracket pattern (`grep "[t]rain_asr"`), or use the PID |
| a successful `/health` with the GPU idle: a wrong `GGML_VK_VISIBLE_DEVICES` silently fell back to CPU | confirm GPU use (`nvidia-smi`) or check the server's offload lines |
| metrics computed in RAM and written only at the end, then lost to a kill | write and `fsync` per line, and support `--resume` (the eval scripts already do) |
| `timeout` wrapped around a benchmark that only prints at the end: a kill looks like a clean exit | no `timeout` on long jobs; use tmux |
| results appended to an old file looked like racing writes | open outputs with `"w"`, or check for stale rows |
| a benchmark run against a stale phone binary | `md5sum` host vs phone after every push |
| the llama-server host prompt cache (8 GiB default) thrashed the phone | `--cache-ram 0` for ASR |
| an uncooled phone under constant load produced NaN output from the NPU | cooling, pacing, and a thermal guard (keep it below about 90 °C) |
| generating 100k WAVs inside an agent session doubled disk use via snapshots | generate bulk data outside agent sessions |

## Open items (good first tasks)

1. **Spontaneous Vietnamese ASR data**: the biggest known quality gap.
2. **Neural VAD (Silero) + live mic input** in `harness/src/mono.cc`. The energy VAD will not hold up in noisy rooms.
3. Port the per-language segmenting policy into `mono.cc`.
4. NaN guard on NPU ASR (non-finite first logits → re-run on GPU), then a cooled 30-minute soak test.
5. Parametrise the few remaining hard-coded paths (`ASR/configs/*.json` data paths are relative to `ASR/`; check script headers).
6. A working Korean TTS voice (`ko_KR-kss-medium` needs pygoruut, which Piper 1.2 ignores).
7. Fix the trainer so intermediate checkpoints include tokenizer/processor files.
8. Upstream the llama.cpp conversion fixes in `patches/`.
