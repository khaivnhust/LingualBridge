# patches

`llama.cpp-conversion.patch` — local fixes to the `llama.cpp` submodule (pinned at `25747b08e`) that the
Qwen3-ASR GGUF export depends on (WORKLOG §8):

- `conversion/base.py`: the tokenizer probe-hash check rejects any trimmed vocabulary (bug #1).
- `conversion/qwen3vl.py`: Qwen3-ASR conversion fixes for `rope.dimension_sections`, the top-level
  `audio_config`, and the audio-tower bias mapping (bugs #2–#4).

Apply after cloning:

```bash
git submodule update --init
git -C llama.cpp apply ../patches/llama.cpp-conversion.patch
```

Bug #5 (the chat template is silently overwritten during conversion) is **not** fixed here; the pipeline
sends the ASR prompt explicitly instead.
