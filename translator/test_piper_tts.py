#!/bin/python3
"""
Piper TTS Test Script
Test Piper TTS models interactively or with built-in test sentences.

Usage:
    Interactive mode:
        python3 test_piper_tts.py

    Test all sentences for a language:
        python3 test_piper_tts.py --lang en
        python3 test_piper_tts.py --lang zh
        python3 test_piper_tts.py --lang ko
        python3 test_piper_tts.py --lang vi

    Test specific text:
        python3 test_piper_tts.py --text "Hello world"

    Test all languages:
        python3 test_piper_tts.py --all
"""

import os
import sys
import time
import json
import subprocess
import argparse

PIPER_DIR = os.path.expanduser("~/AI-Translator/piper")
PIPER_BIN = os.path.join(PIPER_DIR, "install/piper")
ESPEAK_DATA = os.path.join(PIPER_DIR, "install/espeak-ng-data")
VOICES_DIR = os.path.join(PIPER_DIR, "voices")
TEST_SENTENCES_DIR = os.path.join(PIPER_DIR, "etc/test_sentences")
OUTPUT_DIR = os.path.expanduser("~/AI-Translator/translator/piper_outputs")

MODELS = {
    "en": os.path.join(VOICES_DIR, "en_US-lessac-medium.onnx"),
    "zh": os.path.join(VOICES_DIR, "zh_CN-huayan-medium.onnx"),
    "ko": os.path.join(VOICES_DIR, "ko_KR-kss-medium.onnx"),
    "vi": os.path.join(VOICES_DIR, "vi_VN-vais1000-medium.onnx"),
}

TEST_FILES = {
    "en": os.path.join(TEST_SENTENCES_DIR, "test_en-us.jsonl"),
    "zh": os.path.join(TEST_SENTENCES_DIR, "test_zh-cn.jsonl"),
    "vi": os.path.join(TEST_SENTENCES_DIR, "test_vi.jsonl"),
}

SAMPLE_TEXTS = {
    "en": "Hello, this is a test of the Piper text to speech system.",
    "zh": "你好，这是Piper语音合成系统的测试。",
    "ko": "안녕하세요, 파이퍼 텍스트 음성 변환 시스템의 테스트입니다.",
    "vi": "Xin chào, đây là bài kiểm tra hệ thống chuyển đổi văn bản thành giọng nói Piper.",
}

KOREAN_SENTENCES = [
    "인공지능은 컴퓨터 과학의 한 분야로, 지능적인 기계를 만드는 것을 목표로 한다.",
    "오늘 날씨가 정말 좋습니다. 공원에서 산책하기에 완벽한 날이에요.",
    "한국어는 대한민국의 공용어이며, 전 세계에 약 팔천만 명이 사용합니다.",
    "기술의 발전으로 우리의 일상생활이 많이 변화하고 있다.",
    "음성 합성 기술은 자연스러운 목소리를 컴퓨터로 만들어내는 기술이다.",
]


def run_piper(text, model_path, output_path):
    """Run piper and return timing info."""
    cmd = [
        PIPER_BIN,
        "--model", model_path,
        "--output_file", output_path,
        "--espeak_data", ESPEAK_DATA,
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd, input=text, capture_output=True, text=True, timeout=30
    )
    elapsed = time.time() - t0

    # Parse RTF from piper output
    rtf = None
    for line in result.stderr.split("\n"):
        if "Real-time factor:" in line:
            try:
                rtf = float(line.split("Real-time factor:")[1].split("(")[0].strip())
            except (ValueError, IndexError):
                pass

    # Get file size
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return {
        "elapsed": elapsed,
        "rtf": rtf,
        "file_size": file_size,
        "output_path": output_path,
        "error": result.stderr if result.returncode != 0 else None,
    }


def test_sentence(text, lang, output_dir, index=0):
    """Test a single sentence."""
    model_path = MODELS[lang]
    output_file = os.path.join(output_dir, f"{lang}_{index:03d}.wav")

    info = run_piper(text, model_path, output_file)

    if info["error"]:
        print(f"    ERROR: {info['error'][:100]}")
        return None

    rtf_str = f"{info['rtf']:.3f}x" if info['rtf'] else "N/A"
    speed_str = f"{1/info['rtf']:.1f}x" if info['rtf'] and info['rtf'] > 0 else "N/A"
    size_str = f"{info['file_size']/1024:.0f}KB"

    print(f"    RTF={rtf_str} ({speed_str} realtime)  {size_str}  {info['output_path']}")
    return info


def test_lang(lang, num_sentences=None):
    """Test all sentences for a language."""
    print(f"\n{'='*60}")
    print(f"Language: {lang.upper()}")
    print(f"Model: {MODELS[lang]}")
    print(f"{'='*60}")

    output_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(output_dir, exist_ok=True)

    sentences = []

    # Load from test file
    if lang in TEST_FILES and os.path.exists(TEST_FILES[lang]):
        with open(TEST_FILES[lang], "r") as f:
            for line in f:
                data = json.loads(line.strip())
                sentences.append(data["text"])

    # Add Korean sentences if no test file
    if lang == "ko" and not sentences:
        sentences = KOREAN_SENTENCES

    # Add sample text if no sentences
    if not sentences:
        sentences = [SAMPLE_TEXTS.get(lang, "Hello world")]

    if num_sentences:
        sentences = sentences[:num_sentences]

    print(f"  Testing {len(sentences)} sentences...\n")

    results = []
    total_time = 0
    total_rtf = []

    for i, text in enumerate(sentences):
        print(f"  [{i+1}/{len(sentences)}] {text[:60]}...")
        info = test_sentence(text, lang, output_dir, i)
        if info:
            results.append(info)
            total_time += info["elapsed"]
            if info["rtf"] and info["rtf"] > 0:
                total_rtf.append(info["rtf"])

    if results:
        avg_rtf = sum(total_rtf) / len(total_rtf) if total_rtf else 0
        print(f"\n  Summary: {len(results)} sentences, avg RTF={avg_rtf:.3f}x ({1/avg_rtf:.1f}x realtime), total {total_time:.2f}s")


def interactive_mode():
    """Interactive TTS testing."""
    print("Piper TTS Interactive Test")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Output: {OUTPUT_DIR}")
    print("\nCommands:")
    print("  Type text to synthesize (auto-detects language)")
    print("  ':lang en' to switch language")
    print("  ':list' to list output files")
    print("  ':quit' to exit\n")

    lang = "en"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counter = 0

    while True:
        try:
            text = input(f"[{lang.upper()}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue

        if text == ":quit":
            break

        if text == ":list":
            for f in sorted(os.listdir(OUTPUT_DIR)):
                size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
                print(f"  {f} ({size/1024:.0f}KB)")
            continue

        if text.startswith(":lang "):
            new_lang = text.split()[1]
            if new_lang in MODELS:
                lang = new_lang
                print(f"  Switched to {lang.upper()}")
            else:
                print(f"  Unknown language. Available: {', '.join(MODELS.keys())}")
            continue

        output_file = os.path.join(OUTPUT_DIR, f"interactive_{lang}_{counter:03d}.wav")
        counter += 1

        info = run_piper(text, MODELS[lang], output_file)
        if info["error"]:
            print(f"  ERROR: {info['error'][:100]}")
        else:
            rtf_str = f"RTF={info['rtf']:.3f}x" if info['rtf'] else ""
            print(f"  -> {info['output_path']} ({info['file_size']/1024:.0f}KB) {rtf_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Piper TTS Test")
    parser.add_argument("--lang", choices=list(MODELS.keys()), help="Language to test")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--all", action="store_true", help="Test all languages")
    parser.add_argument("--num", type=int, default=None, help="Number of sentences to test")
    args = parser.parse_args()

    if args.text:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        lang = args.lang or "en"
        output_file = os.path.join(OUTPUT_DIR, "custom.wav")
        print(f"Text: {args.text}")
        print(f"Language: {lang.upper()}")
        info = run_piper(args.text, MODELS[lang], output_file)
        if info["error"]:
            print(f"ERROR: {info['error']}")
        else:
            rtf_str = f"RTF={info['rtf']:.3f}x ({1/info['rtf']:.1f}x realtime)" if info['rtf'] else ""
            print(f"Output: {info['output_path']} ({info['file_size']/1024:.0f}KB) {rtf_str}")
    elif args.all:
        for lang in MODELS:
            test_lang(lang, args.num)
    elif args.lang:
        test_lang(args.lang, args.num)
    else:
        interactive_mode()
