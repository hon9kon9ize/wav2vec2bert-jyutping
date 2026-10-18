"""How much of the error is a character that legitimately has several readings (particles such as 呢, 啦, 嘅, ...)?

    python particle_analysis.py [--preds tmp/preds_run21.json] [--extra_path <dir with ToJyutping>] [--min_share 0.03]

For every held-out clip the Chinese text is aligned with the label syllables (one syllable per character). The readings
each character has in the 300,886 training labels are counted; a reading that covers at least --min_share of the
character's uses is "attested". Errors are then split into: the model gave another attested reading of the same character
(reading variant: the label is one of several acceptable answers) and everything else. Also lists the characters with the
most errors and what the labellers used for them.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def main():
    os.environ.setdefault("USE_TF", "0")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=os.path.join(REPO, "tmp", "preds_run21.json"))
    ap.add_argument("--extra_path", default=None)
    ap.add_argument("--min_share", type=float, default=0.03)
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    args = ap.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)
    import ToJyutping as tj

    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from error_consistency import align
    from lexical_experiment import CJK, build_lexicon

    _, r1, _, _, _ = build_lexicon(tj, args.emo_path)                 # char -> Counter(label syllable)
    sp = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)
    emo = load_any_dataset(args.emo_path)["test"]
    texts = {"tts_val": sp["validation"]["text"], "tts_test": sp["test"]["text"], "emo_test": emo["text"]}
    groups = {"tts": ("tts_val", "tts_test"), "emo": ("emo_test",)}
    attested = {c: {s for s, n in cnt.items() if n / sum(cnt.values()) >= args.min_share} for c, cnt in r1.items()}
    multi = {c for c, a in attested.items() if len(a) >= 2}
    print(f"characters in the lexicon: {len(r1)}; with 2 or more attested readings (>= {100 * args.min_share:.0f}% of their uses): {len(multi)}")

    data = json.load(open(args.preds))
    for g, sets in groups.items():
        syl = err = var = multi_syl = multi_err = 0
        by_char_err, by_char_n, by_char_var = Counter(), Counter(), Counter()
        wrong_as = defaultdict(Counter)
        for c in data:
            if c["set"] not in sets:
                continue
            chars = [ch for ch in texts[c["set"]][c["idx"]] if CJK.search(ch)]
            if len(chars) != len(c["gold"]):
                continue
            for op, i, r, h in align(c["gold"], c["pred"]):
                if op not in ("=", "S", "D"):
                    continue
                ch = chars[i]
                syl += 1; by_char_n[ch] += 1
                if ch in multi:
                    multi_syl += 1
                if op != "=":
                    err += 1; by_char_err[ch] += 1; wrong_as[ch][h if op == "S" else "(deleted)"] += 1
                    if ch in multi:
                        multi_err += 1
                    if op == "S" and h in attested.get(ch, ()):
                        var += 1; by_char_var[ch] += 1
        print("\n" + "=" * 100)
        print(f"{g} held-out: {syl} syllables, {err} errors ({100 * err / syl:.2f}% SER)")
        print(f"  characters with several attested readings are {100 * multi_syl / syl:.1f}% of the syllables but {100 * multi_err / max(err, 1):.0f}% of the errors "
              f"(error rate {100 * multi_err / max(multi_syl, 1):.2f}% vs {100 * (err - multi_err) / max(syl - multi_syl, 1):.2f}% for the rest)")
        print(f"  errors where the model's answer is ANOTHER ATTESTED READING of the same character: {var} of {err} ({100 * var / max(err, 1):.0f}%) "
              f"-> reading-tolerant SER {100 * (err - var) / syl:.2f}% (strict {100 * err / syl:.2f}%)")
        print("  characters with the most errors (errors / uses; labellers' readings in training; what the model said instead):")
        for ch, e in by_char_err.most_common(12):
            cnt = r1.get(ch, Counter()); tot = max(sum(cnt.values()), 1)
            reads = ", ".join(f"{s} {100 * n / tot:.0f}%" for s, n in cnt.most_common(4))
            inst = ", ".join(f"{s}:{n}" for s, n in wrong_as[ch].most_common(3))
            print(f"     {ch}  {e:3}/{by_char_n[ch]:4}  labels: {reads:48} model: {inst}  [attested-variant errors {by_char_var[ch]}]")


if __name__ == "__main__":
    main()
