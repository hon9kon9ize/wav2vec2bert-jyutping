"""Patterns in the tone-merger errors (tone 3/6, 4/6, 2/5), following Mok & Wong (2010), "Production and perception of the
rising tones in Hong Kong Cantonese": mergers are individual (speakers differ), not categorical, directional in places, and
(in two speakers) commoner in high-frequency words.

    python tone_merger_patterns.py [--preds tmp/preds_run21.json]      # predictions from `error_consistency.py decode`

Three checks on the model's held-out errors (tts validation + test, emo test):
  1. direction of each merger pair (3->6 versus 6->3, ...);
  2. do merger-pair tone errors cluster within clips (as they would if a speaker or an utterance merges consistently)?
     compared with the same number of errors scattered at random over the same syllables;
  3. is the merger-pair error rate higher for frequent syllables?
"""

import argparse
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import TONE_MERGERS, align, parse  # noqa: E402


def merger_tone_error(r, h):
    po, pn, pc, pt = parse(r)
    ho, hn, hc, ht = parse(h)
    return (po, pn, pc) == (ho, hn, hc) and pt != ht and frozenset((pt, ht)) in TONE_MERGERS


def training_syllable_counts(emo_path):
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text

    cnt = Counter()
    tr = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select_columns(["phone"])
    for p in tr["phone"]:
        cnt.update(normalize_jyutping_text(p).split())
    emo = load_any_dataset(emo_path)
    emo = emo["train"] if hasattr(emo, "keys") else emo
    for p in emo.select_columns(["labels"])["labels"]:
        cnt.update(normalize_jyutping_text(p).split())
    return cnt


def main():
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=os.path.join(REPO, "tmp", "preds_run21.json"))
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    args = ap.parse_args()
    data = json.load(open(args.preds))
    groups = {"tts held-out": ("tts_val", "tts_test"), "emo held-out": ("emo_test",)}
    counts = training_syllable_counts(args.emo_path)
    rng = np.random.default_rng(0)

    for gname, sets in groups.items():
        clips = [c for c in data if c["set"] in sets]
        per_clip, direction, sylls = [], Counter(), []          # sylls: (clip index, frequency, is merger-tone error)
        for ci, c in enumerate(clips):
            k = 0
            for op, _, r, h in align(c["gold"], c["pred"]):
                if op not in ("=", "S"):
                    continue
                e = op == "S" and merger_tone_error(r, h)
                k += e
                if e:
                    direction[(parse(r)[3], parse(h)[3])] += 1
                sylls.append((ci, counts.get(r, 0), e))
            per_clip.append(k)
        n_err = sum(per_clip)
        print("=" * 100)
        print(f"{gname}: {len(clips)} clips, {len(sylls)} syllables, {n_err} merger-pair tone errors ({100 * n_err / len(sylls):.2f}% of syllables)")

        print("\n1. Direction (gold tone -> model tone: count; the reverse direction in brackets)")
        done = set()
        for (a, b), v in sorted(direction.items(), key=lambda kv: -kv[1]):
            if frozenset((a, b)) in done:
                continue
            done.add(frozenset((a, b)))
            print(f"   T{a}->T{b}: {v}   (T{b}->T{a}: {direction.get((b, a), 0)})")

        print("\n2. Clustering of the errors within clips (clips with 2 or more merger-pair tone errors)")
        obs = sum(1 for k in per_clip if k >= 2)
        ci_of = np.array([s[0] for s in sylls])
        sim = []
        for _ in range(2000):
            pick = rng.choice(len(sylls), size=n_err, replace=False)
            sim.append(sum(1 for v in np.bincount(ci_of[pick], minlength=len(clips)) if v >= 2))
        print(f"   observed {obs} clips; expected if the same number of errors were scattered at random over the syllables: "
              f"{np.mean(sim):.1f} (95% range {np.percentile(sim, 2.5):.0f}-{np.percentile(sim, 97.5):.0f}); "
              f"P(random >= observed) = {np.mean([x >= obs for x in sim]):.3f}")

        print("\n3. Merger-pair tone error rate by how often the gold syllable occurs in the training transcripts")
        buckets = [(0, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9)]
        for lo, hi in buckets:
            sel = [s for s in sylls if lo <= s[1] < hi]
            e = sum(s[2] for s in sel)
            print(f"   {lo:>6}-{hi if hi < 10 ** 9 else 'inf':<6} syllables {len(sel):6}  merger-pair tone errors {e:4}  rate {100 * e / max(len(sel), 1):.2f}%")
        print()


if __name__ == "__main__":
    main()
