"""How close are the dataset labels to ToJyutping(text)? (label provenance check)

    pip install ToJyutping          # or: pip install --target <dir> ToJyutping  and pass --tojyutping_path <dir>
    python check_labels_vs_tojyutping.py [--tojyutping_path <dir>] [--preds tmp/preds_run21.json]

Both datasets carry the Chinese `text` next to the jyutping label. For every clip the text is converted with ToJyutping
and compared, syllable by syllable, with the label. Optionally (--preds, from `error_consistency.py decode`) the model's
errors on the held-out clips are then split by whether the label agrees with the converter.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import ONSET_MERGERS, TONE_MERGERS, align, parse  # noqa: E402

CJK = re.compile(r"[一-鿿]")
SYL = re.compile(r"[a-z]+[1-6]")


def convert(tj, text):
    """text -> (converter syllables, number of CJK characters the converter could not read)"""
    syls, missing = [], 0
    for ch, jp in tj.get_jyutping_list(text):
        if jp:
            syls.extend(SYL.findall(jp))
        elif CJK.search(ch):
            missing += 1
    return syls, missing


def part_diff(a, b):
    """parts that differ between two syllables, and whether they are all merger pairs"""
    po, pn, pc, pt = parse(a)
    ho, hn, hc, ht = parse(b)
    parts = [n for n, d in (("onset", po != ho), ("rime", pn != hn or pc != hc), ("tone", pt != ht)) if d]
    merger = ("rime" not in parts) and ("onset" not in parts or frozenset((po or "-", ho or "-")) in ONSET_MERGERS) \
        and ("tone" not in parts or frozenset((pt, ht)) in TONE_MERGERS)
    return parts, merger


def load_texts(emo_path):
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split

    out = {}
    sp = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)
    for name, key in (("tts_train", "train"), ("tts_val", "validation"), ("tts_test", "test")):
        d = sp[key].select_columns(["text", "phone"])
        out[name] = (d["text"], d["phone"])
    emo = load_any_dataset(emo_path)
    for name, key in (("emo_train", "train"), ("emo_test", "test")):
        d = emo[key].select_columns(["text", "labels"])
        out[name] = (d["text"], d["labels"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tojyutping_path", default=None)
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--preds", default=os.path.join(REPO, "tmp", "preds_run21.json"))
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    from jyutping import normalize_jyutping_text

    data = load_texts(args.emo_path)
    conv = {}
    print("=" * 100)
    print("1. Labels vs ToJyutping(text), per split")
    print(f"{'split':10} {'clips':>7} {'unreadable':>11} {'same length':>12} {'identical':>10} {'syllable agreement':>19}   differing syllables: onset / rime / tone / merger-only")
    for name, (texts, labels) in data.items():
        n = same_len = ident = unread = 0
        tot = agree = 0
        diff_parts, merger, nsub, pairs = Counter(), 0, 0, Counter()
        conv[name] = []
        for t, l in zip(texts, labels):
            c, miss = convert(tj, t)
            g = normalize_jyutping_text(l).split()
            conv[name].append(c)
            n += 1
            unread += miss > 0
            same_len += len(c) == len(g)
            ident += c == g
            ops = align(g, c)
            tot += len(g)
            agree += sum(op == "=" for op, *_ in ops)
            for op, _, r, h in ops:
                if op == "S":
                    nsub += 1
                    parts, mg = part_diff(r, h)
                    for p in parts:
                        diff_parts[p] += 1
                    merger += mg
                    pairs[(r, h)] += 1
        print(f"{name:10} {n:>7} {100 * unread / n:>10.1f}% {100 * same_len / n:>11.1f}% {100 * ident / n:>9.1f}% {100 * agree / tot:>18.2f}%   "
              f"{diff_parts['onset']} / {diff_parts['rime']} / {diff_parts['tone']} / {merger} of {nsub} substitutions")
        print(f"{'':10} most common label -> converter: " + ", ".join(f"{a}>{b}:{c}" for (a, b), c in pairs.most_common(8)))

    if not os.path.exists(args.preds):
        return
    preds = json.load(open(args.preds))
    print("\n" + "=" * 100)
    print("2. Where the model disagrees with the label (clips where label and converter have the same number of syllables)")
    print("   L = label, C = ToJyutping, P = model prediction")
    for fam, sets in (("tts held-out", ("tts_val", "tts_test")), ("emo held-out", ("emo_test",))):
        cat = Counter()
        n_pos = n_clips = n_used = 0
        merger_err = Counter()
        for s in sets:
            for c in (p for p in preds if p["set"] == s):
                n_clips += 1
                cs = conv[s][c["idx"]]
                gold, pred = c["gold"], c["pred"]
                if len(cs) != len(gold):
                    continue
                n_used += 1
                ops = align(gold, pred)
                for op, i, r, h in ops:
                    if op not in ("=", "S"):
                        continue
                    n_pos += 1
                    cv = cs[i]
                    if r == cv and op == "=":
                        cat["L=C, model right (agrees with both)"] += 1
                    elif r == cv:
                        cat["L=C, model wrong (deviates from the canonical reading)"] += 1
                        merger_err["canon"] += part_diff(r, h)[1]
                    elif op == "=":
                        cat["L!=C, model reproduces the label"] += 1
                    elif h == cv:
                        cat["L!=C, model wrong but sides with the converter"] += 1
                        merger_err["conv"] += part_diff(r, h)[1]
                    else:
                        cat["L!=C, model differs from both"] += 1
                        merger_err["both"] += part_diff(r, h)[1]
        lc = sum(v for k, v in cat.items() if k.startswith("L!=C"))
        errs = sum(v for k, v in cat.items() if "wrong" in k or "differs" in k)
        print(f"\n  [{fam}] {n_used} of {n_clips} clips usable, {n_pos} syllables; label differs from the converter at {lc} ({100 * lc / n_pos:.2f}%)")
        for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
            print(f"     {v:6}  {k}")
        print(f"     model errors: {errs}; of these the label equals the converter in {cat['L=C, model wrong (deviates from the canonical reading)']} "
              f"({100 * cat['L=C, model wrong (deviates from the canonical reading)'] / max(errs, 1):.0f}%)")
        if lc:
            print(f"     where the label deviates from the converter, the model reproduces the label {100 * cat['L!=C, model reproduces the label'] / lc:.0f}% of the time, "
                  f"sides with the converter {100 * cat['L!=C, model wrong but sides with the converter'] / lc:.0f}%")
        print(f"     merger-pair-only errors by case: {dict(merger_err)}")


if __name__ == "__main__":
    main()
