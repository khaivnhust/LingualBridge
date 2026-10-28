# Multilingual Speech-to-Speech Translation Pipeline

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  Mic Input   │───▶│  VAD (WebRT) │───▶│  ASR (Whisper│───▶│  MT (mBART-50│
│  sounddevice │    │  webrtcvad   │    │  multilingual)│    │  or NLLB-200)│
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐                        ┌─────────────┐
│  Speaker Out│◀───│  TTS (Piper) │◀───────────────────────│  EN Text    │
│  sounddevice│    │  per-language │                        └─────────────┘
└─────────────┘    └──────────────┘
```

## Component Choices

### 1. ASR: Whisper Multilingual (Option A)

**Model:** `openai/whisper-base` (74M params, 150MB)

**Why Whisper:**
- ✅ Code-switching support ("我那个GPUが高すぎた" → detects mixed ZH/JA)
- ✅ Auto language detection
- ✅ Single model for all 4 input languages
- ✅ Vocab trimming feasible
- ✅ Distillation tools exist (Distil-Whisper)

**Latency:** ~1.5-2s on RPi5 (base), ~2-3s on Orange Pi 5 Pro

**Alternatives considered:**
- Moonshine: Too small, can't handle code-switching
- Faster-Whisper: Good option, but Whisper base is the fallback

### 2. Translation: mBART-50-multilingual (Option A)

**Model:** `facebook/mbart-large-50-many-to-many-mmt` (610M params, 2.2GB)

**Why mBART-50:**
- ✅ Direct 4-language support (EN, ZH, KO, VN)
- ✅ Handles code-switched input
- ✅ Mature, well-tested
- ✅ Can be fine-tuned for domain-specific vocab

**Latency:** ~1-2s on edge devices (with quantization)

**Alternatives considered:**
- NLLB-200: Higher quality, but 3.3B params too large for edge
- NLLB-200-distilled: 600M params, good fallback if mBART-50 quality insufficient
- MarianMT: Fast, but weaker on code-switching

### 3. TTS: Piper (per target language)

**Models:** Language-specific voices from Rhasspy/Piper

| Language | Model | Size | Quality |
|----------|-------|------|---------|
| EN | `en_US-amy-medium` | 63MB | Excellent |
| ZH | `zh_CN-huayan-medium` | 200MB | Good |
| KO | `ko_KR-very_seonghoon-medium` | 110MB | Good |
| VN | `vi_VN-vais1000-medium` | 110MB | Good |

**Total TTS models:** ~483MB

**Latency:** ~0.5-1s on RPi5

**Why Piper:**
- ✅ Supports all 4 target languages
- ✅ Fast enough for real-time on edge
- ✅ Local inference (no API calls)
- ✅ Multiple voice quality options

## Full Pipeline Specs

### Model Sizes

| Component | Model | Params | Size (disk) | RAM |
|-----------|-------|--------|-------------|-----|
| ASR | whisper-base | 74M | 150MB | 300MB |
| MT | mBART-50 | 610M | 2.2GB | 2.5GB |
| TTS (EN) | piper en | - | 63MB | 100MB |
| TTS (ZH) | piper zh | - | 200MB | 300MB |
| TTS (KO) | piper ko | - | 110MB | 150MB |
| TTS (VN) | piper vn | - | 110MB | 150MB |
| **TOTAL** | | 684M | **~2.8GB** | **~3.5GB** |

### Latency Breakdown

| Stage | Latency (RPi5) | Latency (Orange Pi 5) |
|-------|----------------|----------------------|
| VAD | 10ms | 10ms |
| ASR | 1.5-2.5s | 1-1.5s |
| MT | 1-2s | 0.5-1s |
| TTS | 0.5-1s | 0.3-0.8s |
| **TOTAL** | **3-5s** | **2-3s** |

### Accuracy Expectations

| Language Pair | Expected Quality | Notes |
|---------------|------------------|-------|
| EN→ZH | ★★★★☆ | Best supported |
| EN→KO | ★★★★☆ | Good support |
| EN→VN | ★★★☆☆ | Slightly weaker |
| ZH→EN | ★★★★☆ | Strong |
| ZH→KO | ★★★☆☆ | May need fine-tuning |
| ZH→VN | ★★☆☆☆ | Weakest, may need custom training |

## Implementation Phases

### Phase 1: Core Pipeline (Week 1-2)

**Goal:** Working prototype on edge device

1. **ASR Setup**
   - Install whisper-base with word-level timestamps
   - Add code-switching prompt guidance
   - Implement language detection fallback

2. **MT Setup**
   - Install mBART-50 with INT8 quantization
   - Add EN/ZH/KO/VN translation pairs
   - Implement code-switched text handling

3. **TTS Setup**
   - Install piper with all 4 language voices
   - Add prosody alignment for code-switched text

4. **Integration**
   - Create `pipeline_core.py` with WebRTC VAD
   - Add streaming support
   - Create `run_pipeline.sh` launch script

**Deliverables:**
- `pipeline_core.py` - Main pipeline script
- `run_pipeline.sh` - Launch script
- `requirements.txt` - Dependencies
- Test with 10 sample sentences per language

### Phase 2: Optimization (Week 3-4)

**Goal:** Reduce latency and memory

1. **ASR Optimization**
   - Vocab trimming to 4 languages
   - Quantization to INT8
   - Whisper distillation (optional)

2. **MT Optimization**
   - Knowledge distillation to smaller model
   - INT8 quantization
   - Batch inference optimization

3. **Pipeline Optimization**
   - Async processing between stages
   - Model preloading
   - GPU acceleration (if available)

**Deliverables:**
- Optimized models
- Benchmark results
- Memory usage report

### Phase 3: Fine-tuning (Week 5-6)

**Goal:** Domain-specific accuracy

1. **Data Collection**
   - Gather 100+ hours of target domain audio
   - Create parallel text data
   - Domain-specific vocabulary lists

2. **Model Fine-tuning**
   - Fine-tune ASR on target domain
   - Fine-tune MT on domain-specific vocab
   - Adjust TTS for domain terminology

3. **Evaluation**
   - BLEU scores for translation quality
   - WER for ASR accuracy
   - User satisfaction testing

**Deliverables:**
- Fine-tuned models
- Evaluation report
- Performance comparison

## Code-Switching Support

### Input Examples

```
"我那个GPUが高すぎた" → ZH+JA mixed
"The API needs Transformer模型" → EN+ZH mixed  
"GPU가 고장났어요" → KR mixed (English tech terms)
"Cần update API cho dự án" → VN mixed (English tech terms)
```

### Handling Strategy

1. **ASR:** Whisper handles natively (no special handling needed)
2. **MT:** mBART-50 can process mixed text, outputs single language
3. **TTS:** Use dominant language voice, fall back to English for foreign words

### Translation Examples

```
Input: "我那个GPUが高すぎ了"
ASR: "我那个GPU太高了" (corrected to ZH)
MT (→EN): "My GPU is too expensive"
MT (→KO): "내 GPU가 너무 비싸요"
MT (→VN): "GPU của tôi quá đắt"
```

## Device-Specific Configurations

### Raspberry Pi 5 (4GB)

```yaml
components:
  asr:
    model: whisper-base
    device: cpu
    quantization: int8
    batch_size: 1
    chunk_size: 30s
  
  mt:
    model: mbart-large-50
    device: cpu
    quantization: int8
    max_length: 512
  
  tts:
    model: piper
    device: cpu
    voice: language-specific
    speed: 1.0

memory_limit: 3.5GB
swap: 2GB
```

### Orange Pi 5 Pro (16GB)

```yaml
components:
  asr:
    model: whisper-base
    device: npu (RKNN)
    quantization: int8
    batch_size: 4
    chunk_size: 60s
  
  mt:
    model: mbart-large-50
    device: cpu
    quantization: int8
    max_length: 1024
  
  tts:
    model: piper
    device: npu
    voice: language-specific
    speed: 1.0

memory_limit: 8GB
swap: 4GB
npu_acceleration: true
```

## Deployment Structure

```
/home/albert/AI-Translator/translator/
├── PIPELINE_PLAN.md          # This file
├── requirements.txt          # Python dependencies
├── run_pipeline.sh           # Launch script
├── pipeline_core.py          # Main pipeline
├── components/
│   ├── asr/
│   │   ├── whisper_asr.py    # ASR component
│   │   └── models/           # Whisper models
│   ├── mt/
│   │   ├── mbart_mt.py       # Translation component
│   │   └── models/           # mBART models
│   └── tts/
│       ├── piper_tts.py      # TTS component
│       └── voices/           # Piper voices
├── utils/
│   ├── vad.py                # Voice Activity Detection
│   ├── audio.py              # Audio I/O utilities
│   └── text.py               # Text processing
├── tests/
│   ├── test_asr.py           # ASR tests
│   ├── test_mt.py            # MT tests
│   ├── test_tts.py           # TTS tests
│   └── test_pipeline.py      # Integration tests
├── benchmarks/
│   ├── latency_test.py       # Latency measurement
│   └── accuracy_test.py      # Accuracy evaluation
└── data/
    ├── samples/              # Test audio samples
    └── vocab/                # Domain vocabulary
```

## Success Criteria

### Minimum Viable Product (Phase 1)

- [ ] ASR handles all 4 input languages
- [ ] MT translates EN ↔ ZH/KO/VN
- [ ] TTS outputs in all 4 target languages
- [ ] Total latency < 6 seconds
- [ ] Memory usage < 4GB
- [ ] Works on RPi5 (4GB)

### Production Ready (Phase 2)

- [ ] Total latency < 3 seconds
- [ ] Memory usage < 3GB
- [ ] Handles code-switching correctly
- [ ] Graceful error handling
- [ ] Auto language detection

### Optimized (Phase 3)

- [ ] Fine-tuned models for domain
- [ ] BLEU score > 25 for all pairs
- [ ] User satisfaction > 80%
- [ ] Works on Arduino UNO Q (2GB)

## Open Questions

1. **Training data availability:** Do you have parallel audio-text data for fine-tuning?
2. **Target domain:** What specific domain/terminology should the system handle?
3. **Priority:** Which language pair is most important to optimize first?
4. **Voice preferences:** Any specific voice styles for TTS output?

## Next Steps

1. **Start Phase 1** - Create working prototype
   ```bash
   pip install transformers sounddevice webrtcvad piper-tts
   # Test each component individually
   # Integrate into pipeline
   ```

2. **Test with sample data** - Verify accuracy
   ```bash
   python pipeline_core.py --test
   ```

3. **Deploy to edge device** - Initial testing
   ```bash
   scp pipeline_core.py pi@raspberrypi:~/
   ssh pi@raspberrypi "python pipeline_core.py"
   ```

4. **Gather feedback** - Iterate on accuracy
   - Test with real users
   - Collect error cases
   - Fine-tune models

---

**Created:** 2026-01-26  
**Author:** AI-Translator Team  
**Status:** Phase 1 Planning Complete
