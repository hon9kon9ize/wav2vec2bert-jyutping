"""Does the model treat the merger pairs as categorical or as 'between two tones'?  (Mok & Wong 2010, Table 3)

    python tone_ambiguity.py [--feats tmp/pitch_feats.json]     # from `check_tone_pitch.py build` (tone posteriors per syllable)

Mok & Wong code each token of a target tone as produced as the target, as the partner tone, or 'between the two'. Their two
female speakers (the most confusing of 17 potential mergers) produced T2 as T5 in 14% of 90 tokens (13/90) and between the
two in 2%, T5 as T2 in 4% of 72 (3/72) and between in 4% (3/72), T5 as T3/T6 in 3% (2/72). Here, for every held-out syllable
whose gold tone belongs to a merger pair and whose rime is unchanged, q = p(gold tone) / (p(gold tone) + p(partner tone)) is
taken from the model's tone posterior at the rime's peak frame, and the syllable is called 'gold' (q >= 0.9), 'between'
(0.1 < q < 0.9) or 'partner' (q <= 0.1).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import parse  # noqa: E402

PARTNERS = {"2": ["5"], "5": ["2"], "3": ["6"], "6": ["3", "4"], "4": ["6"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", default=os.path.join(REPO, "tmp", "pitch_feats.json"))
    args = ap.parse_args()
    blob = json.load(open(args.feats))
    recs, clips = blob["recs"], blob["clips"]
    groups = {"tts held-out": ("tts_val", "tts_test"), "emo held-out": ("emo_test",)}
    print(f"{'':14} {'gold -> partner':16} {'n':>6} {'gold':>7} {'between':>8} {'partner':>8}   | where the model's tone is the partner's: share of gold-tone syllables")
    for gname, sets in groups.items():
        cnt = defaultdict(lambda: [0, 0, 0, 0])
        for r in recs:
            if r["set"] not in sets:
                continue
            g = r["L"][-1]
            if g not in PARTNERS or parse(r["L"])[:3] != parse(r["P"])[:3]:
                continue
            tp = clips[f"{r['set']}:{r['idx']}"]["tp"][r["j"]]
            if g not in tp:
                continue
            for h in PARTNERS[g]:
                if h not in tp:
                    continue
                pg, ph = 2.718281828 ** tp[g], 2.718281828 ** tp[h]
                q = pg / (pg + ph)
                c = cnt[(g, h)]
                c[0] += 1
                c[1 if q >= 0.9 else 3 if q <= 0.1 else 2] += 1
        for (g, h), c in sorted(cnt.items()):
            n = c[0]
            print(f"{gname:14} T{g} -> T{h}:{'':8} {n:>6} {100 * c[1] / n:6.1f}% {100 * c[2] / n:7.1f}% {100 * c[3] / n:7.1f}%")
        print()
    print("Mok & Wong, two merging female speakers (rates per token): T2 -> T5 14% (13/90), T2 between 2% (2/90); "
          "T5 -> T2 4% (3/72), T5 between 4% (3/72), T5 -> T3/T6 3% (2/72)")


if __name__ == "__main__":
    main()
