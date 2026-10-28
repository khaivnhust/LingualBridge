#!/bin/python3
import sys
import os
import time
import tempfile
import base64
import io
import wave
import struct
import ctypes
import numpy as np
import sounddevice as sd
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Gemma4ChatHandler,
    ImmutableSandboxedEnvironment,
    Jinja2ChatFormatter,
    _grammar_for_response_format,
    suppress_stdout_stderr,
)
import llama_cpp
import llama_cpp.llama_types as llama_types
import llama_cpp.mtmd_cpp as mtmd_cpp
import jinja2
from typing import List, Optional, Dict, Any, Union, Iterator

# --- Config ---
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
CHUNK_DURATION = 0.5
SILENCE_THRESHOLD = 0.015
SILENCE_TIMEOUT = 1.2
MAX_RECORD_SECONDS = 30
MIN_RECORD_SECONDS = 0.3

# --- Args ---
target = sys.argv[1] if len(sys.argv) > 1 else "VN"
gguf_path = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/"
    "fc034cfff751157913579611efad8462ac1be606/gemma-4-12b-it-Q4_K_M.gguf"
)
mmproj_path = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/"
    "fc034cfff751157913579611efad8462ac1be606/mmproj-F16.gguf"
)


class AudioGemma4ChatHandler(Gemma4ChatHandler):
    """Gemma4ChatHandler patched to also handle input_audio content parts."""

    def __call__(self, *, llama, messages, functions=None, function_call=None,
                 tools=None, tool_choice=None, temperature=0.2, top_p=0.95,
                 top_k=40, min_p=0.05, typical_p=1.0, stream=False, stop=[],
                 seed=None, response_format=None, max_tokens=None,
                 presence_penalty=0.0, frequency_penalty=0.0, repeat_penalty=1.1,
                 tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1,
                 model=None, logits_processor=None, grammar=None, logit_bias=None,
                 logprobs=None, top_logprobs=None, **kwargs):

        self._init_mtmd_context(llama)
        assert self.mtmd_ctx is not None

        media_marker = self._mtmd_cpp.mtmd_default_marker().decode("utf-8")

        # Extract ALL media (images + audio) in message order
        all_media = self._extract_media_from_messages(messages)

        template_env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
            extensions=[Jinja2ChatFormatter.IgnoreGenerationTags, jinja2.ext.loopcontrols],
        )
        template_env.filters["tojson"] = Jinja2ChatFormatter.tojson
        template = template_env.from_string(self._get_chat_template(llama))

        def raise_exception(message):
            raise ValueError(message)

        text = template.render(
            messages=self._get_template_messages(messages, media_marker),
            add_generation_prompt=True,
            eos_token=self._decode_token_piece(llama.detokenize([llama.token_eos()])),
            bos_token=self._decode_token_piece(llama.detokenize([llama.token_bos()])),
            raise_exception=raise_exception,
            functions=functions, function_call=function_call,
            tools=tools, tool_choice=tool_choice,
            strftime_now=Jinja2ChatFormatter.strftime_now,
            **kwargs,
        )

        # Replace image URLs with media markers in rendered text
        for media in all_media:
            if media["type"] == "image_url":
                text = text.replace(media["url"], media_marker)

        if self.verbose:
            print(text, file=sys.stderr)

        # Create bitmaps in message order (images AND audio)
        bitmaps = []
        bitmap_cleanup = []
        try:
            for media in all_media:
                if media["type"] == "image_url":
                    image_bytes = self.load_image(media["url"])
                    bitmap = self._create_bitmap_from_bytes(image_bytes)
                    bitmaps.append(bitmap)
                    bitmap_cleanup.append(bitmap)
                elif media["type"] == "audio":
                    audio_bitmap = self._create_audio_bitmap(media["samples"])
                    if audio_bitmap is not None:
                        bitmaps.append(audio_bitmap)
                        bitmap_cleanup.append(audio_bitmap)

            input_text = self._mtmd_cpp.mtmd_input_text()
            input_text.text = text.encode("utf-8")
            input_text.add_special = True
            input_text.parse_special = True

            chunks = self._mtmd_cpp.mtmd_input_chunks_init()
            if chunks is None:
                raise ValueError("Failed to create input chunks")

            try:
                bitmap_array = (self._mtmd_cpp.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
                result = self._mtmd_cpp.mtmd_tokenize(
                    self.mtmd_ctx, chunks, ctypes.byref(input_text),
                    bitmap_array, len(bitmaps),
                )

                if result != 0:
                    raise ValueError(f"Failed to tokenize input: error code {result}")

                llama.reset()
                llama._ctx.kv_cache_clear()

                n_chunks = self._mtmd_cpp.mtmd_input_chunks_size(chunks)

                for i in range(n_chunks):
                    chunk = self._mtmd_cpp.mtmd_input_chunks_get(chunks, i)
                    if chunk is None:
                        continue

                    chunk_type = self._mtmd_cpp.mtmd_input_chunk_get_type(chunk)

                    if chunk_type == self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_TEXT:
                        n_tokens_out = ctypes.c_size_t()
                        tokens_ptr = self._mtmd_cpp.mtmd_input_chunk_get_tokens_text(
                            chunk, ctypes.byref(n_tokens_out)
                        )
                        if tokens_ptr and n_tokens_out.value > 0:
                            tokens = [tokens_ptr[j] for j in range(n_tokens_out.value)]
                            if llama.n_tokens + len(tokens) > llama.n_ctx():
                                raise ValueError(
                                    f"Prompt exceeds n_ctx: {llama.n_tokens + len(tokens)} > {llama.n_ctx()}"
                                )
                            llama.eval(tokens)

                    elif chunk_type in [
                        self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                        self._mtmd_cpp.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    ]:
                        chunk_n_tokens = self._mtmd_cpp.mtmd_input_chunk_get_n_tokens(chunk)
                        if llama.n_tokens + chunk_n_tokens > llama.n_ctx():
                            raise ValueError(
                                f"Prompt exceeds n_ctx: {llama.n_tokens + chunk_n_tokens} > {llama.n_ctx()}"
                            )

                        new_n_past = llama_cpp.llama_pos(0)
                        result = self._mtmd_cpp.mtmd_helper_eval_chunk_single(
                            self.mtmd_ctx, llama._ctx.ctx, chunk,
                            llama_cpp.llama_pos(llama.n_tokens),
                            llama_cpp.llama_seq_id(0), llama.n_batch,
                            False, ctypes.byref(new_n_past),
                        )
                        if result != 0:
                            raise ValueError(f"Failed to evaluate chunk: error code {result}")
                        llama.n_tokens = new_n_past.value

                prompt = llama.input_ids[: llama.n_tokens].tolist()

            finally:
                self._mtmd_cpp.mtmd_input_chunks_free(chunks)

        finally:
            for bitmap in bitmap_cleanup:
                self._mtmd_cpp.mtmd_bitmap_free(bitmap)

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        if functions is not None:
            tools = [{"type": "function", "function": function} for function in functions]
        if function_call is not None:
            if isinstance(function_call, str) and function_call in ("none", "auto"):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {"type": "function", "function": {"name": function_call["name"]}}

        completion = llama.create_completion(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, min_p=min_p, typical_p=typical_p,
            logprobs=logprobs, stream=stream,
            stop=stop, seed=seed, presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty, repeat_penalty=repeat_penalty,
            tfs_z=tfs_z, mirostat_mode=mirostat_mode, mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta, model=model, logits_processor=logits_processor,
            grammar=grammar, logit_bias=logit_bias,
        )

        if stream:
            return completion

        return {
            "id": completion["id"],
            "object": "chat.completion",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [
                {
                    "index": choice["index"],
                    "message": {
                        "role": "assistant",
                        "content": choice["text"],
                    },
                    "logprobs": choice.get("logprobs"),
                    "finish_reason": choice["finish_reason"],
                }
                for choice in completion["choices"]
            ],
            "usage": completion["usage"],
        }

    @staticmethod
    def _convert_content_part_for_template(part, media_marker):
        if isinstance(part, dict) and part.get("type") == "image_url":
            return {"type": "text", "text": media_marker}
        if isinstance(part, dict) and part.get("type") in ("audio", "input_audio"):
            return {"type": "text", "text": media_marker}
        return part

    @staticmethod
    def _extract_media_from_messages(messages):
        """Extract image URLs and audio data in message order."""
        all_media = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "image_url":
                    url_data = part.get("image_url")
                    if isinstance(url_data, dict) and "url" in url_data:
                        all_media.append({"type": "image_url", "url": url_data["url"]})
                    elif isinstance(url_data, str):
                        all_media.append({"type": "image_url", "url": url_data})
                elif ptype in ("audio", "input_audio"):
                    audio_data = part.get("input_audio", part.get("audio", {}))
                    if isinstance(audio_data, dict):
                        b64 = audio_data.get("data", "")
                        fmt = audio_data.get("format", "wav")
                        if b64:
                            samples = _decode_audio_b64(b64, fmt)
                            if samples is not None and len(samples) > 0:
                                all_media.append({"type": "audio", "samples": samples})
        return all_media

    def _create_audio_bitmap(self, samples_float32):
        """Create mtmd_bitmap from float32 audio samples."""
        if self.mtmd_ctx is None:
            raise ValueError("mtmd context not initialized")

        if not self._mtmd_cpp.mtmd_support_audio(self.mtmd_ctx):
            print("  (warning: mtmd does not support audio)", file=sys.stderr)
            return None

        n_samples = len(samples_float32)
        audio_rate = self._mtmd_cpp.mtmd_get_audio_sample_rate(self.mtmd_ctx)
        if audio_rate <= 0:
            print("  (warning: invalid audio sample rate)", file=sys.stderr)
            return None

        # Resample if needed
        if audio_rate != SAMPLE_RATE:
            ratio = audio_rate / SAMPLE_RATE
            new_len = int(n_samples * ratio)
            indices = np.linspace(0, n_samples - 1, new_len)
            resampled = np.interp(indices, np.arange(n_samples), samples_float32).astype(np.float32)
        else:
            resampled = samples_float32.astype(np.float32)

        c_float_array = ctypes.c_float * len(resampled)
        audio_ptr = c_float_array(*resampled.tolist())

        bitmap = self._mtmd_cpp.mtmd_bitmap_init_from_audio(len(resampled), audio_ptr)
        if bitmap is None:
            print("  (warning: failed to create audio bitmap)", file=sys.stderr)
            return None

        return bitmap


def _decode_audio_b64(b64_data, fmt="wav"):
    """Decode base64 audio to float32 samples."""
    try:
        raw = base64.b64decode(b64_data)
        wav_buf = io.BytesIO(raw)
        with wave.open(wav_buf, "rb") as w:
            frames = w.readframes(w.getnframes())
            sample_width = w.getsampwidth()
            if sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
            else:
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            if w.getnchannels() > 1:
                audio = audio[::w.getnchannels()]
            return audio
    except Exception as e:
        print(f"  (warning: failed to decode audio: {e})", file=sys.stderr)
        return None


print(f"Loading model: {gguf_path}")
print(f"Loading mmproj: {mmproj_path}")

handler = AudioGemma4ChatHandler(mmproj_path, verbose=False)

llm = Llama(
    model_path=gguf_path,
    n_ctx=4096,
    n_gpu_layers=35,
    n_batch=512,
    chat_handler=handler,
    verbose=False,
)
print("Model loaded.\n")

# --- Language prompts ---
LANG_PROMPTS = {
    "EN": "Translate the following speech into English. Output ONLY the English translation. No explanations.",
    "ZH": "将以下语音翻译成中文。只输出中文翻译。不要解释。保留技术术语的英文原文（如 GPU、API）。",
    "KR": "다음 음성을 한국어로 번역하세요. 한국어 번역만 출력합니다. 설명하지 마세요. 전문용어는 영어 원어를 유지합니다.",
    "VN": "Dịch giọng nói sau sang tiếng Việt. Chỉ xuất ra bản dịch tiếng Việt. Không giải thích. Giữ nguyên thuật ngữ kỹ thuật bằng tiếng Anh.",
}

print(f"\n🎙️  Voice translator ready! Target: {target}")
print(f"   Speak into your microphone. Say '!reset' or press Ctrl+C to exit.\n")


def record_utterance():
    chunks = []
    silence_start = None
    speaking_started = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
    ) as stream:
        while True:
            data, overflowed = stream.read(stream.blocksize)
            audio_chunk = data[:, 0]
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

            if rms > SILENCE_THRESHOLD:
                speaking_started = True
                silence_start = None
                chunks.append(audio_chunk.copy())
            elif speaking_started:
                chunks.append(audio_chunk.copy())
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT:
                    break

            total_duration = len(chunks) * CHUNK_DURATION
            if total_duration > MAX_RECORD_SECONDS:
                break

    if not chunks:
        return None

    audio = np.concatenate(chunks)
    total_sec = len(audio) / SAMPLE_RATE

    if total_sec < MIN_RECORD_SECONDS:
        return None

    return audio


def translate_audio(audio_path, target_lang):
    prompt = LANG_PROMPTS.get(target_lang, LANG_PROMPTS["VN"])

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    result = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ],
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
    )

    return result["choices"][0]["message"]["content"].strip()


# --- Main loop ---
cumulative_speech = ""

while True:
    try:
        print("🎤 Listening... (speak now)")
        audio = record_utterance()

        if audio is None:
            print("   (no speech detected)\n")
            continue

        duration = len(audio) / SAMPLE_RATE
        print(f"   Captured {duration:.1f}s of audio. Translating...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import scipy.io.wavfile as wavfile
            wavfile.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
            tmp_path = f.name

        try:
            translation = translate_audio(tmp_path, target)
        finally:
            os.unlink(tmp_path)

        if not translation:
            print("   (empty translation)\n")
            continue

        if "!reset" in translation.lower():
            cumulative_speech = ""
            print("✨ History cleared.\n")
            continue

        cumulative_speech = f"{cumulative_speech} {translation}".strip() if cumulative_speech else translation
        print(f"👉 {translation}\n")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
