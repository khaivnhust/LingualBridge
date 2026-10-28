#!/usr/bin/env python3
"""Paired bootstrap CI on the conversational-English WER delta (Round 1 vs Round 4).

Reuses score_hyps.py's own normalisation and manifest loading so the point
estimate is identical to `score_hyps.py --hyps ... --ref-manifest ...`; only the
uncertainty estimate is new.

Bootstrap is over UTTERANCES (the resampling unit), not characters or tokens,
because utterances are the independent observations. Corpus WER is recomputed as
sum(errors)/sum(ref_tokens) within each resample, which is the standard corpus
bootstrap -- resampling per-utterance *rates* would over-weight short utterances.
"""
import argparse
import json
import random
import sys
import unicodedata

import jiwer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import score_hyps as SH


def counts(ref, hyp, lang="en"):
    """(errors, ref_len) for one utterance; errors = S + D + I.

    score_hyps normalises by stripping ALL spaces for zh/ko, so those languages
    are scored character-wise (CER) and en/vi word-wise (WER). Must match
    score_hyps.METRIC or the point estimate will not reproduce.
    """
    charwise = SH.METRIC.get(lang) == "cer"
    if not ref:
        n = len(hyp) if charwise else len(hyp.split())
        return (float(n) if hyp.strip() else 0.0), 0
    proc = jiwer.process_characters if charwise else jiwer.process_words
    o = proc([ref], [hyp])
    return float(o.substitutions + o.deletions + o.insertions), float(o.hits + o.substitutions + o.deletions)


def load_aligned(hyp_a, hyp_b, manifest, lang="en"):
    """Aligned per-utterance (ref, hyp_a, hyp_b) on the intersection of keys."""
    refs = SH.load_refs(manifest=manifest)
    # load_hyps returns (dict, key_used) -- take the dict.
    A, B = SH.load_hyps(hyp_a)[0], SH.load_hyps(hyp_b)[0]
    keys = sorted(set(A) & set(B) & set(refs))
    rows = []
    for k in keys:
        r, lg = refs[k]
        if lg != lang:
            continue
        rows.append((SH.normalize(r, lg), SH.normalize(A[k]["hyp"], lg), SH.normalize(B[k]["hyp"], lg)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-a", required=True, help="baseline hyps (Round 1)")
    ap.add_argument("--hyp-b", required=True, help="candidate hyps (Round 4)")
    ap.add_argument("--ref-manifest", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_aligned(a.hyp_a, a.hyp_b, a.ref_manifest, a.lang)
    n = len(rows)
    if n == 0:
        sys.exit("no aligned utterances")

    ca = [counts(r, ha, a.lang) for r, ha, _ in rows]
    cb = [counts(r, hb, a.lang) for r, _, hb in rows]

    def wer(c):
        e = sum(x[0] for x in c)
        d = sum(x[1] for x in c)
        return 100.0 * e / d if d else 0.0

    wa, wb = wer(ca), wer(cb)
    delta = wb - wa

    rng = random.Random(a.seed)
    deltas = []
    for _ in range(a.iters):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [ca[i] for i in idx]
        sb = [cb[i] for i in idx]
        deltas.append(wer(sb) - wer(sa))
    deltas.sort()

    def pct(p):
        return deltas[min(len(deltas) - 1, max(0, int(round(p * (len(deltas) - 1)))))]

    lo, hi = pct(0.025), pct(0.975)
    p_better = 100.0 * sum(1 for d in deltas if d < 0) / len(deltas)

    # Paired sign test on per-utterance error-count deltas (distribution-free).
    wins = sum(1 for i in range(n) if cb[i][0] < ca[i][0])
    loss = sum(1 for i in range(n) if cb[i][0] > ca[i][0])
    ties = n - wins - loss

    print(f"  n={n} utterances, {a.iters} bootstrap resamples, seed {a.seed}")
    print(f"  A  {a.hyp_a.split('/')[-1]:<26} {a.lang.upper()}  {wa:6.2f}")
    print(f"  B  {a.hyp_b.split('/')[-1]:<26} {a.lang.upper()}  {wb:6.2f}")
    print(f"  delta (B - A)                 {delta:+6.2f}")
    print(f"  95% CI                        [{lo:+.2f}, {hi:+.2f}]")
    print(f"  P(B better)                   {p_better:5.1f}%")
    print(f"  per-utt  B better {wins}, worse {loss}, tied {ties}")
    verdict = "SIGNIFICANT" if hi < 0 else ("SIGNIFICANT (B worse)" if lo > 0 else "NOT SIGNIFICANT")
    print(f"  verdict                       {verdict}")


if __name__ == "__main__":
    main()