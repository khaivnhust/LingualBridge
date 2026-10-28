#!/bin/python3
"""
Test Gemma 4 12B audio transcription ability on FLEURS dataset.
Uses GGUF + mmproj via llama-cpp-python with custom AudioGemma4ChatHandler.
"""

import os
import sys
import time
import base64
import io
import wave
import ctypes
import numpy as np
import jiwer
from datasets import load_dataset
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    Gemma4ChatHandler,
    ImmutableSandboxedEnvironment,
    Jinja2ChatFormatter,
    _grammar_for_response_format,
)
import llama_cpp
import llama_cpp.llama_types as llama_types
import llama_cpp.mtmd_cpp as mtmd_cpp
import jinja2

SAMPLE_RATE = 16000


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

        for media in all_media:
            if media["type"] == "image_url":
                text = text.replace(media["url"], media_marker)

        if self.verbose:
            print(text, file=sys.stderr)

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


# --- Audio encoding ---

def audio_to_wav_b64(audio_np, sr=16000):
    audio_int16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# --- Load Gemma ---

def load_gemma(gguf_path, mmproj_path, n_gpu_layers=35):
    print(f"  Loading mmproj: {mmproj_path}")
    handler = AudioGemma4ChatHandler(mmproj_path, verbose=False)

    print(f"  Loading model: {gguf_path}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        chat_handler=handler,
        verbose=False,
    )
    return llm


def transcribe_gemma(llm, audio_np, sr=16000, prompt=None):
    if prompt is None:
        prompt = "Transcribe the following speech exactly. Output ONLY the transcription text. No explanations."

    audio_b64 = audio_to_wav_b64(audio_np, sr)

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
        temperature=0.0,
    )

    return result["choices"][0]["message"]["content"].strip()


# --- FLEURS loading ---

FLEURS_MAP = {
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
    "vi": "vi_vn",
}

def load_fleurs_samples(lang_config, num_samples, split="validation"):
    ds = load_dataset("google/fleurs", lang_config, split=split, streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        ref = item["transcription"]
        samples.append({"audio": audio, "sr": sr, "ref": ref})
    return samples


# --- Main ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 Audio Transcription Test")
    parser.add_argument("--langs", nargs="+", default=["en", "zh", "ko", "vi"],
                        choices=list(FLEURS_MAP.keys()))
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--gguf", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/"
        "fc034cfff751157913579611efad8462ac1be606/gemma-4-12b-it-Q4_K_M.gguf"))
    parser.add_argument("--mmproj", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/"
        "fc034cfff751157913579611efad8462ac1be606/mmproj-F16.gguf"))
    parser.add_argument("--n-gpu-layers", type=int, default=35)
    args = parser.parse_args()

    print("Gemma 4 12B Audio Transcription Test")
    print(f"Languages: {args.langs}")
    print(f"Samples per language: {args.num_samples}")
    print(f"GPU layers: {args.n_gpu_layers}")
    print()

    llm = load_gemma(args.gguf, args.mmproj, args.n_gpu_layers)
    print("Model loaded.\n")

    results = {}

    for lang in args.langs:
        lang_config = FLEURS_MAP[lang]
        print(f"{'='*60}")
        print(f"Language: {lang.upper()} ({lang_config}) — {args.num_samples} samples")
        print(f"{'='*60}")

        print("  Loading dataset...")
        samples = load_fleurs_samples(lang_config, args.num_samples)
        print(f"  Loaded {len(samples)} samples\n")

        hyps = []
        refs = []
        total_audio_sec = 0
        t_start = time.time()

        for i, sample in enumerate(samples):
            audio = sample["audio"]
            sr = sample["sr"]
            ref = sample["ref"]
            audio_dur = len(audio) / sr
            total_audio_sec += audio_dur

            t0 = time.time()
            try:
                hyp = transcribe_gemma(llm, audio, sr)
            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
                hyp = ""
            elapsed = time.time() - t0
            rtf = elapsed / audio_dur if audio_dur > 0 else 0

            hyps.append(hyp)
            refs.append(ref)

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{i+1}/{len(samples)}] RTFx={1/rtf:.1f}x  hyp={hyp[:70]}")

        total_time = time.time() - t_start
        avg_rtf = total_time / total_audio_sec if total_audio_sec > 0 else 0

        use_cer = lang in ("zh", "ko")
        metric_fn = jiwer.cer if use_cer else jiwer.wer
        metric_name = "CER" if use_cer else "WER"

        scores = []
        for hyp, ref in zip(hyps, refs):
            try:
                s = metric_fn(ref, hyp)
                scores.append(s)
            except Exception:
                pass
        avg_score = np.mean(scores) * 100 if scores else -1

        key = f"gemma4_{lang}"
        results[key] = {
            "model": "gemma-4-12b-it-Q4_K_M",
            "lang": lang,
            "metric": metric_name,
            "score": avg_score,
            "rtfx": 1 / avg_rtf if avg_rtf > 0 else 0,
            "vram_mb": 0,
            "num_samples": len(samples),
            "total_audio_sec": total_audio_sec,
        }

        print(f"\n  RESULT: {metric_name}={avg_score:.2f}%  RTFx={1/avg_rtf:.1f}x")

    # Summary
    print(f"\n{'='*60}")
    print("GEMMA 4 TRANSCRIPTION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Lang':<6} {'Metric':<6} {'Score%':<10} {'RTFx':<10}")
    print(f"{'-'*6} {'-'*6} {'-'*10} {'-'*10}")
    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['lang']:<6} {r['metric']:<6} {r['score']:<10.2f} {r['rtfx']:<10.1f}")

    all_scores = [r["score"] for r in results.values()]
    all_rtfx = [r["rtfx"] for r in results.values()]
    print(f"\n{'AVG':<6} {'':<6} {np.mean(all_scores):<10.2f} {np.mean(all_rtfx):<10.1f}")
