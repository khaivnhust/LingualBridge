# LingualBridge: TTS Voice Selection & Fine-tuning Plan
 
**Author:** Lâm (ML Optimization) · **Date:** 18 Oct 2026 · **Status:** voices selected and integrated into `pipeline.py`; fine-tuning planned but not started
 
## TL;DR
 
- We listened to **23 candidate voices** (official and community Piper, plus sherpa-onnx VITS) and picked one voice per language:
 
  | Lang | Voice | License |
  |---|---|---|
  | VI | **CakeByVPBank Piper v3, speaker 2 "Quang Huy"** (male, Northern accent) | MIT ⚠️ |
  | EN | **`en_US-ryan-high`** (male) | **CC BY-NC-SA 4.0** |
  | ZH | **`zh_CN-huayan-medium`** (female) | unknown |
  | KO | **YJin1983 `piper_ko_female_v1`** (female) | not stated |
 
  VI ⚠️: the label says MIT, but the voice was trained on data from a CC-BY-NC teacher (see §4).
 
- All four use espeak-ng phonemes and run **10–50× faster than real time on a laptop CPU**, using about 200–300 MB RAM each.
- **3 of 4 licenses are non-commercial or unclear.** This is fine for the competition but needs to be asked at the 23/10 workshop.
- **Code-switching test:** English technical terms (CNC, PLC, sensor, conveyor…) are **silently dropped** by lexicon-based Chinese voices. espeak voices keep them but read them with English phonemes. One clear bug: Vietnamese espeak spells *jig* as "gi i đê".
- **Recommendation for MT:** prompt Hy-MT2 to **keep technical terms unchanged** and **write symbols as words**. If we do this, **we must not do the "knowledge drop" step** from `MT_DISTILLATION.md`.
- **Next:** fine-tune our own 4 voices with clean licenses (teacher VoxCPM2, Apache-2.0), running unattended on the dev laptop. This takes about 8 days and does not block anything else. The current voices remain the fallback.
 
## 1. Selected voices
 
| Lang | Voice | File (`piper/voices/`) | Size | RTF\* | Peak RAM\*\* | License |
|---|---|---|---|---|---|---|
| VI | CakeByVPBank Piper v3, **speaker_id 2** (Quang Huy) | `vi_VN-csa-voice-piper-v3-medium.onnx` | 77.1 MB | 0.032 | 217 MB | MIT ⚠️ see §4 |
| EN | `en_US-ryan-high` | `en_US-ryan-high.onnx` | 120.8 MB | 0.083 | 312 MB | **CC BY-NC-SA 4.0** |
| ZH | `zh_CN-huayan-medium` | `zh_CN-huayan-medium.onnx` | 63.2 MB | 0.020 | 236 MB | unknown |
| KO | YJin1983 `piper_ko_female_v1` | `piper_ko_female_v1.onnx` | 63.5 MB | 0.026 | 241 MB | not stated |
 
\* Real-time factor on the dev laptop CPU (AMD Ryzen 7 7840H). **These numbers were not measured on the phone.** Method: 1 warm-up sentence, then 3 factory sentences × 2 runs; RTF = synthesis time / audio duration.
\*\* Peak RSS of the Python process after 9 sentences, including the Python runtime. The TTS part alone is about 175–271 MB.
 
All voices output 22,050 Hz mono. Integration:
- `translator/pipeline.py`: `PIPER_MODELS`, `PIPER_SPEAKERS = {"vi": 2}`, `tts_piper()`
- `translator/test_piper_tts.py`
 
**Important change:** TTS now uses the **Python `piper-tts` library (1.6.0)** instead of the C++ binary `piper/install/piper`. The KO voice config uses multi-codepoint phonemes (`aɪ aʊ ɔɪ eɪ oʊ`), and the old binary crashes on load with `"aɪ" is not a single codepoint`. Voices are now loaded once and cached. `tts_piper()` keeps the same return format, so `pipeline_test.py` did not need changes.
 
### License-safe alternatives (if the organizers say no to NC licenses)
- EN: `en_US-libritts_r-medium` (CC BY 4.0, 904 speakers) or `en_GB-cori-high` (public domain)
- ZH: `chaowen` (CC0). It needs a transliteration step first, because it drops all Latin words (§3).
- KO: no clean option available, so this is the main reason to fine-tune (§6).
 
## 2. How we compared
 
- **Factory set:** 3 sentences with the same meaning in all 4 languages (safety warning, shift announcement, technical question).
- **Code-switching set:** 6 sentences × 4 languages, each containing English technical terms: CNC, PLC / sensor, conveyor / motor, inverter / QC, mold, jig / robot, manual / PPE, SOP.
- **Engines:** `piper-tts` 1.6.0 (espeak voices), `sherpa-onnx` 1.13.8 (the runtime we propose for Android), and onnxruntime directly for SNAP.
- Loudness was normalized for fair listening, because sherpa-onnx does not normalize its output (peak 0.58–0.77).
- **Listening pages** (local):
  - `translator/piper_outputs/compare/index.html` (42 clips)
  - `translator/piper_outputs/compare_codeswitch/index.html` (252 clips)
- **Reproduce:** `python3 translator/compare_tts_voices.py [--set codeswitch] [--lang vi|en|zh|ko]`. Note: running with `--lang` overwrites `index.html` with only that language.
 
## 3. Results: all 23 voices
 
Android: ✅ = works with a standard runtime (Piper / sherpa-onnx + espeak). ⚠️ = needs an extra component.
 
### Vietnamese
| Voice | Source | License | MB | RTF | Android |
|---|---|---|---|---|---|
| **CakeByVPBank v3** (5 speakers: Ngọc Lan, Minh Anh, **Quang Huy**, Thu Hà, Yến Nhi) ✅ selected | Community | MIT | 77.1 | 0.032 | ✅ |
| vi_VN-vais1000-medium (previous) | Official | CC BY 4.0 | 63.2 | 0.020 | ✅ |
| vnhellboy tramvang / diemdam / altotram | Community | other (lessac lineage, research only) | 63.5 | 0.022–0.027 | ✅ |
| ACut giong_nu_pho_thong / thanh_nien_tu_tin | Community | other | 63.5 | 0.020 | ✅ |
 
### English
| Voice | Source | License | MB | RTF | Android |
|---|---|---|---|---|---|
| **en_US-ryan-high** ✅ selected | Official | CC BY-NC-SA 4.0 | 120.8 | 0.083 | ✅ |
| en_US-lessac-medium (previous) | Official | Blizzard 2013 (research) | 63.2 | 0.019 | ✅ |
| en_US-amy-medium | Official | see mimic3-voices | 63.2 | 0.019 | ✅ |
| en_GB-cori-high | Official | public domain | 114.2 | 0.089 | ✅ |
| en_US-libritts_r-medium (904 speakers) | Official | CC BY 4.0 | 78.6 | 0.020 | ✅ |
| OpenVoiceOS miro en-US | Community | CC BY-NC-ND 4.0 | 63.5 | 0.019 | ✅ |
 
### Chinese
| Voice | Source | License | Phonemizer | MB | RTF | Android |
|---|---|---|---|---|---|---|
| **zh_CN-huayan-medium** ✅ selected | Official | unknown | espeak cmn | 63.2 | 0.020 | ✅ |
| chaowen 超文 | Newer Piper voice (sherpa build) | CC0 | pinyin + lexicon | 63.2 | 0.028 | ✅ |
| xiao_ya 小雅 | Newer Piper voice (sherpa build) | non-commercial (Baker) | pinyin + lexicon | 63.2 | 0.030 | ✅ |
| colafly yt-chinese_female | Community | not stated | espeak cmn | 63.5 | 0.021 | ✅ |
 
### Korean
| Voice | Source | License | Phonemizer | MB | RTF | Android |
|---|---|---|---|---|---|---|
| **YJin1983 piper_ko_female_v1** ✅ selected | Community | not stated | espeak ko | 63.5 | 0.026 | ✅ (see §4) |
| SNAP (22 speakers: reading / kind styles, male + female) | Community | Apache-2.0 (AI Hub data) | jamo + **cloud** G2P | 121.2 | 0.119–0.125 | ⚠️ needs a Korean G2P |
| vits-mimic3-ko_KO-kss_low | sherpa-onnx release | CC BY-NC-SA (KSS) | espeak | 62.8 | 0.034 | ✅ APK exists |
| ko_KR-kss-medium (previous) | Community | CC BY-NC-SA (KSS) | pygoruut | 63.5 | — | ⚠️ |
 
"Medium" voices run at RTF ≈ 0.02–0.03. "High" voices and SNAP are about 4× slower.
 
## 4. Findings and gotchas
 
1. **Lexicon-based Chinese voices drop every Latin word.** sherpa-onnx logs `Ignore OOV` for chaowen and xiao_ya. All 13 technical terms in all 6 test sentences were dropped. For example, "motor的inverter过热了" becomes "的过热了", so the listener no longer knows which device is overheating.
2. **espeak keeps English terms but converts them to English phonemes**, even in VI/ZH/KO voices that were trained on native phonemes. Each voice needs a listening check on the terms. One definite bug: Vietnamese espeak spells *jig* as "gi i đê". Acronyms are read with Vietnamese letter names (CNC → "xê en nờ xê", QC → "quy xê"), which sounds natural to Vietnamese listeners.
3. **SNAP Korean is not really offline.** Its demo calls a cloud API for G2P and part-of-speech tags, and refuses to synthesize without it. Offline it can only take raw jamo. Its fp16 ONNX export is also broken (`tensor(float16) ... does not match expected type (tensor(float))`).
4. **The YJin1983 Korean voice crashes the old C++ Piper binary** because of multi-codepoint phonemes. The Python library handles them correctly. On Android/sherpa-onnx these phonemes only appear in English words inside Korean text, which is exactly where technical terms go, so they need a check.
5. **Quang Huy's MIT license is questionable.** CakeByVPBank generated its training data with OmniVoice, whose weights are **CC-BY-NC**, and with VoxCPM2.
6. **ryan-high license correction:** an earlier note said CC BY 4.0. The model card actually says **CC BY-NC-SA 4.0** (RyanSpeech dataset).
7. **Speaker IDs are not validated.** A wrong `speaker_id` still produces audio, just in a different voice. Quang Huy is ID 2.
 
## 5. Recommendation for MT: speech-friendly output
 
Goal: **keep technical terms unchanged** and **write symbols as words**. Symbols means °, %, /, ~, ±. Digits can stay, because espeak reads them correctly.
 
Hy-MT2 supports a *Personalization* prompt (see its model card). Fill `{terms}` with **only the glossary terms that actually appear in the source sentence**, to keep the prompt short:
 
```
[Source Text]
{source_text}
 
[Translation Tasks]
1. Keep these technical terms exactly as written, do not translate them: {terms}.
2. Do not use symbols such as °, %, /, ~, ±, ≥. Write them out as spoken words in the target language.
3. Only output the translated result without any additional explanation.
4. Translate the [Source Text] into {target_lang}.
```
 
Constraints from `MT_DISTILLATION.md`:
- **Skip the "knowledge drop" step (§5).** It removes exactly this instruction-following mode.
- **The teacher must generate labels with this same prompt.** Train and deploy must use the same template (§7.4).
- **Measure per direction:** % of terms kept and % of outputs that still contain symbols, in addition to BLEU/chrF.
 
This plan fits the selected voices: all four use espeak and read Latin text.
 
## 6. Fine-tuning plan (summary)
 
Full plan, in Vietnamese: `TTS_FINETUNE_PLAN.md`.
 
**Why:** clean licenses for all 4 voices, technical terms included in the training data, and sentences in the factory domain.
**Where:** the dev laptop (RTX 4060 **8 GB**). It runs unattended and is independent of MT/ASR work. A shared `flock` GPU lock prevents clashes.
 
| Item | Choice |
|---|---|
| Teacher | **VoxCPM2** (`openbmb/VoxCPM2`, Apache-2.0, 2B params, supports VI/KO/ZH/EN, *voice design* from a text description, so no real person is cloned) |
| Teacher fallback | Qwen3-TTS or Fun-CosyVoice3 (Apache-2.0) for KO/ZH/EN. Neither supports VI, so VI would keep the current voice |
| Rejected | OmniVoice, XTTS-v2, F5-TTS, Fish-Speech (non-commercial weights) |
| Base checkpoints | VI: `vais1000/medium` (CC BY 4.0). EN/ZH/KO: `ljspeech/medium` (public domain). Both are in `rhasspy/piper-checkpoints` |
| Output | Piper **medium**, espeak, 22,050 Hz |
| Voice gender | Male for all 4 by default (one consistent "interpreter" voice). ZH/KO male vs female is decided by a noisy-intelligibility test at gate G2 |
 
**Pipeline:**
1. Smoke-test the teacher (VRAM, RTF, CER) → **G1**
2. Design the voices and pick a reference clip → **G2**
3. Build a corpus: ~4,000 sentences per language from OPUS-100, factory-domain sentences, and ~15% code-switched sentences. **No FLEURS or organizer sample data.**
4. Generate audio with the teacher, cloning from the reference clip.
5. Filter by Qwen3-ASR round-trip (keep ≥ 85% of clips).
6. Fine-tune Piper (~0.5–1 day per voice).
7. Export and evaluate → **G7**
8. Integrate.
 
**Ship gate.** A new voice replaces the current one only if **all** of these hold:
- Round-trip WER/CER ≤ current voice and ≤ 5%
- Term recognition ≥ current voice
- Blind A/B preference ≥ 50% and MOS ≥ 3.8
- RTF ≤ 0.04 on the dev CPU and size ≤ 80 MB
 
**Timeline:** about 8 days unattended, about 26–27/10 if started 18/10. That is well before the internal deadline of 12/11.
 
**Main risk:** VoxCPM2 needs about 8 GB VRAM and the GPU has exactly 8 GB. Gate G1 checks this first, and the fallbacks are listed above.
 
## 7. Action items
 
| Who | What | When |
|---|---|---|
| **Khải** | Ask the organizers at the workshop: are non-commercial voice licenses OK (ryan-high CC BY-NC-SA, huayan unknown, YJin1983 unstated)? Ask the same about NLLB | 23/10 |
| **Khải** (K8) | Add a **"pronunciation" column** per language to the glossary (e.g. vi `jig` → "dích") and agree on the term list used by the MT prompt | with K8 (04/11) |
| **Khải / Lâm** | Test the Personalization prompt with Hy-MT2-1.8B on the 6 code-switch sentences × 6 directions. Measure term-kept % and symbol-free % | before K4 label generation |
| **Khải** | **Do not run knowledge drop** if we use the MT prompt above. Generate teacher labels with the same prompt | K4–K5 |
| **Kiên** (N4) | Run the 4 voices on sherpa-onnx Android. Check (a) loudness normalization, (b) KO multi-codepoint phonemes on English terms, (c) Quang Huy = speaker 2 | 30/10 |
| **Lâm** (L2/L7) | Measure RTF and RAM of the 4 voices on the Snapdragon 8 Elite (QDC or the real device) | with L2/L7 |
| **Lâm** | Run the fine-tuning pipeline (§6) once the open decisions below are settled | start ~18–19/10 |
| **Everyone** | Listen and score at gates G2 (pick voices) and G7 (blind A/B of new vs current) | when notified |
 
## 8. Open decisions (need team input)
 
1. **Fine-tuning environment:** separate venv (recommended, protects the working `piper-tts` / `pipeline.py` setup) or the user Python environment?
2. **Male voices for ZH/KO:** OK to decide based on the noise test at G2?
3. **Listeners for G2/G7:** who is available?
4. **Noise for evaluation:** record our own fan and machine noise (more realistic), or use MUSAN (CC BY 4.0)?
 
## 9. Files
 
| File | Content |
|---|---|
| `TTS_REPORT.md` | This report (English) |
| `TTS_VOICE_SELECTION.md` | Same selection report in Vietnamese |
| `TTS_FINETUNE_PLAN.md` | Detailed fine-tuning plan (Vietnamese) |
| `translator/compare_tts_voices.py` | Generates samples and listening pages for all candidates |
| `translator/piper_outputs/compare*/` | 294 sample WAVs + listening pages |
| `piper/voices/` | The 4 selected voices (+ 3 previous ones) |
| `piper/voices/candidates/` | The 17 other candidate models (~1.4 GB). They are still on disk, so the listening pages can be regenerated. Delete them only after the license question is settled |
