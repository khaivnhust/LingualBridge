#!/usr/bin/env python3
"""Spelled-out English numeral -> digit conversion, for OneVoice-style references.

Round 3 (OneVoice) writes numbers as words ("two thousand three hundred and
thirty") while every ASR model writes digits ("2330"). That mismatch alone
produces 60-86% WER on utterances whose content is transcribed perfectly:

    REF: convert two thousand three hundred and thirty from g. m. t. plus ...
    HYP: convert 2330 from gmt 430 to gmt 200          -> 86% WER, content identical

Applied to BOTH reference and hypothesis so the comparison stays symmetric.
Opt-in only -- score_hyps.py never enables this by default, because every other
corpus here (FLEURS, VoxPopuli, Kspon) already writes numbers as digits.

Two forms are handled:
  * numeral phrases  "two thousand three hundred and thirty" -> 2330
  * digit spelling   "six seven two nine seven"             -> 67297  (phone numbers,
    where additive parsing would wrongly give 31)
"""
import re

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}

NUMWORD = set(UNITS) | set(TENS) | set(SCALES)
_SPLIT = re.compile(r"([A-Za-z]+)")

# Loosest possible numeral phrase: any maximal run of number words, with or
# without "and" between them. Used only by strip(), where exact value is
# irrelevant. "and" is part of the run only when two numerals are adjacent, so a
# real conjunction ("one cat AND two dogs") still survives.
_WORD_ALT = ("|".join(sorted(NUMWORD, key=len, reverse=True)))
WORD_LITE = re.compile(
    rf"\b(?:(?:{_WORD_ALT})[\s-]+)*(?:and[\s-]+)?(?:{_WORD_ALT})\b", re.I)


def _parse(words):
    """Evaluate an English numeral phrase to an int, or None if not numeric."""
    lw = [w.lower() for w in words]
    # Digit-spelling form: every token is a bare 0-9 digit -> concatenate.
    if len(lw) > 1 and all(w in UNITS and UNITS[w] < 10 for w in lw):
        return int("".join(str(UNITS[w]) for w in lw))
    total = current = 0
    for w in lw:
        if w == "and":
            continue
        if w in UNITS:
            current += UNITS[w]
        elif w in TENS:
            current += TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in SCALES:
            scale = SCALES[w]
            if scale >= 1000:
                total += (current or 1) * scale
                current = 0
            else:
                current = (current or 1) * scale
        else:
            return None
    # must add the sub-thousand remainder: "two thousand three hundred" = 2300
    return total + current if (total or current) else None


def convert(text):
    """Replace spelled-out numerals in `text` with digits."""
    out, run = [], []

    def flush():
        if not run:
            return
        joined = "".join(run)
        # keep any trailing separator (space/punct) that was swept into the run
        trailing = joined[len(joined.rstrip()):]
        val = _parse(joined.split())
        out.append(joined if val is None else f"{val}{trailing}")
        run.clear()

    for part in _SPLIT.split(text):
        if not part:
            continue
        low = part.lower()
        if low in NUMWORD:
            run.append(part)
        elif low == "and" and run:
            run.append(part)
        elif not part.strip() and run:
            # whitespace inside an active numeral run -- must not break it
            run.append(part)
        else:
            flush()
            out.append(part)
    flush()
    return "".join(out)


def strip(text):
    """Remove ALL numerals -- digits and number words -- from `text`.

    Format-neutral by construction: no choice of numeral style can create or
    hide an error, because neither side carries numbers any more. This is the
    honest way to compare a model that writes "2330" against a reference that
    writes "two thousand three hundred and thirty". Preferred over `convert()`
    when the two sides may disagree about numeral style.
    """
    def repl(m):
        return " "
    # drop digit runs first, then spelled-out numeral phrases
    text = re.sub(r"\d+(?:[.,]\d+)*", repl, text)
    # a spelled-out phrase with no trailing non-numeral word
    text = WORD_LITE.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    cases = [
        "convert two thousand three hundred and thirty from g. m. t. plus four hundred and thirty",
        "seven hundred thirty four main street harper valley oregon six seven two nine seven",
        "the year one thousand nine hundred and ninety",
        "i have one cat and two dogs",
        "no numbers here at all",
        "one thousand nine hundred and ninety and two thousand",
    ]
    for s in cases:
        print(f"  {convert(s)!r}")