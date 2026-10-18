"""Group the model's errors on training clips into types (what kind of error is it, and what would fix it?).

Needs the decodes of `mine_label_disagreements.py decode` and ToJyutping (see check_labels_vs_tojyutping.py):

    python group_errors.py --tojyutping_path <dir> [--min_share 0.03]

Every substitution gets exactly one group, first match wins:
  reading_variant/<part>  the character has several readings in the training labels (each >= --min_share of its uses)
                          and the model gave another attested one: the label is one of several acceptable answers
  merger                  only a merger pair differs (tone 3/6, 4/6, 2/5; onset n/l, ng/-, gw/g) and the reading is unattested
  tone_other / onset_other / rime_other   a single part differs and it is not a merger pair
  multi_part              two or more parts differ (a different syllable, usually a mumbled or unusual word)
Deletions and insertions are counted apart. Writes tmp/label_mining/grouped_errors.jsonl (one row per error).
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from check_labels_vs_tojyutping import part_diff  # noqa: E402
from error_consistency import align  # noqa: E402
from lexical_experiment import char_syllable_pairs  # noqa: E402
from mine_label_disagreements import OUT_DIR, convert  # noqa: E402


def build_lexicon(tj, emo_path, cache):
    """char -> Counter(label syllable) and syllable -> training count, over both corpora's training labels"""
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text

    tts = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select_columns(["text", "phone"])
    emo = load_any_dataset(emo_path)
    emo = (emo["train"] if hasattr(emo, "keys") else emo).select_columns(["text", "labels"])
    by_char, syl_freq = defaultdict(Counter), Counter()
    for texts, labels in ((tts["text"], tts["phone"]), (emo["text"], emo["labels"])):
        for t, label in zip(texts, labels):
            syls = normalize_jyutping_text(label).split()
            syl_freq.update(syls)
            pairs = char_syllable_pairs(tj, t, syls)
            for ch, s in pairs or ():
                by_char[ch][s] += 1
    out = (dict(by_char), syl_freq)
    pickle.dump(out, open(cache, "wb"))
    return out


def group_of(parts, merger, attested_hit):
    if attested_hit:
        return "reading_variant/" + ("+".join(parts) if len(parts) == 1 else "multi")
    if merger:
        return "merger"
    return f"{parts[0]}_other" if len(parts) == 1 else "multi_part"


def collect(clips, tj, by_char, syl_freq, min_share):
    attested = {c: {s for s, n in cnt.items() if n / sum(cnt.values()) >= min_share} for c, cnt in by_char.items()}
    rows, clip_errors = [], Counter()
    for c in clips:
        gold = c["gold"]
        conv, chars, ok = convert(tj, c["text"])
        usable = bool(ok) and len(conv) == len(gold)
        for op, i, g, p in align(gold, c["pred"]):
            if op == "=":
                continue
            clip_errors[c["train_idx"]] += 1
            row = {"train_idx": c["train_idx"], "op": op, "pos": i, "n_syl": len(gold), "dur": c["dur"], "text": c["text"],
                   "gold": g, "pred": p, "final": i is not None and i == len(gold) - 1,
                   "char": chars[i][0] if usable and i is not None else "", "gold_freq": syl_freq.get(g, 0) if g else 0}
            if op == "S":
                parts, merger = part_diff(g, p)
                cv = conv[i] if usable else None
                row.update(parts=parts, merger=merger, conv=cv,
                           cat="no_conv" if cv is None else "agree_conv" if p == cv else "model_only" if g == cv else "all_differ",
                           group=group_of(parts, merger, bool(row["char"]) and p in attested.get(row["char"], ())))
            else:
                row["group"] = "deletion" if op == "D" else "insertion"
            rows.append(row)
    return rows, clip_errors


def examples(rows, k=7):
    top = Counter((r["char"] or "?", r["gold"], r["pred"]) for r in rows).most_common(k)
    return "  ".join(f"{c}:{g}->{p}x{n}" for (c, g, p), n in top)


def report(rows, clips, clip_errors, total_syl):
    n_err = len(rows)
    print(f"{len(clips)} clips, {total_syl} syllables, {n_err} errors ({n_err / total_syl:.2%}): "
          f"{sum(r['op'] == 'S' for r in rows)} substitutions, {sum(r['op'] == 'D' for r in rows)} deletions, "
          f"{sum(r['op'] == 'I' for r in rows)} insertions\n")
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    print(f"{'group':26} {'errors':>6} {'share':>6} {'final':>6} {'rare<1k':>8} {'=ToJy':>6}   top (char:label->model xN)")
    for name, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        subs = [r for r in rs if r["op"] == "S"]
        agree = sum(r["cat"] == "agree_conv" for r in subs) / len(subs) if subs else float("nan")
        rare = sum(r["gold_freq"] < 1000 for r in rs) / len(rs)
        print(f"{name:26} {len(rs):6d} {len(rs) / n_err:6.1%} {sum(r['final'] for r in rs) / len(rs):6.1%} {rare:8.1%} "
              f"{agree:6.0%}   {examples(rs)}")
    print("\n'final' = share at the last syllable of the clip; 'rare<1k' = the label syllable has < 1,000 training uses; "
          "'=ToJy' = share of substitutions where the model equals ToJyutping(text).")

    subs = [r for r in rows if r["op"] == "S"]
    by_char = Counter(r["char"] or "?" for r in subs)
    top = by_char.most_common(20)
    print(f"\nconcentration: {len(by_char)} distinct characters carry the {len(subs)} substitutions; the top 20 carry "
          f"{sum(n for _, n in top) / len(subs):.0%}: " + " ".join(f"{c}{n}" for c, n in top))
    pair = Counter((r["gold"], r["pred"]) for r in subs)
    cum, k = 0, 0
    for _, n in pair.most_common():
        cum += n; k += 1
        if cum >= len(subs) / 2:
            break
    print(f"{len(pair)} distinct label->model syllable pairs; the {k} most frequent make up half of the substitutions")

    hist = Counter(min(n, 5) for n in clip_errors.values())
    in_multi = sum(n for n in clip_errors.values() if n >= 3)
    print(f"\nclips with errors: {len(clip_errors)} of {len(clips)} ({len(clip_errors) / len(clips):.1%}); by error count "
          + ", ".join(f"{'5+' if k == 5 else k}: {hist[k]}" for k in sorted(hist))
          + f"; clips with >= 3 errors hold {in_multi / n_err:.0%} of all errors")
    err_dur = [c["dur"] for c in clips if c["train_idx"] in clip_errors]
    ok_dur = [c["dur"] for c in clips if c["train_idx"] not in clip_errors]
    print(f"mean duration: clips with errors {sum(err_dur) / len(err_dur):.2f} s, without {sum(ok_dur) / len(ok_dur):.2f} s")
    tot_final = len(clips)
    final_err = sum(1 for r in rows if r["final"])
    print(f"error rate at the last syllable {final_err / tot_final:.2%} vs elsewhere {(n_err - final_err) / (total_syl - tot_final):.2%}")
    logf = [math.log10(max(r["gold_freq"], 1)) for r in rows if r["op"] != "I"]
    print(f"median label-syllable training count among errors: {sorted(10 ** x for x in logf)[len(logf) // 2]:,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tojyutping_path", default=None)
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--min_share", type=float, default=0.03)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    clips = [json.loads(line) for line in open(os.path.join(args.out_dir, "decoded.jsonl"))]
    by_char, syl_freq = build_lexicon(tj, args.emo_path, os.path.join(args.out_dir, "lexicon.pkl"))
    rows, clip_errors = collect(clips, tj, by_char, syl_freq, args.min_share)
    report(rows, clips, clip_errors, sum(len(c["gold"]) for c in clips))
    with open(os.path.join(args.out_dir, "grouped_errors.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
