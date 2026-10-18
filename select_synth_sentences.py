"""Pick training sentences for synthetic-data generation, aimed at the model's weak spots (rare syllables).

    python select_synth_sentences.py --out_dir tmp/synth_plan [--n 6000] [--first 500]

Candidate pools: rows >= 6,000 of the tts20250516 TRAIN split (the light runs use rows 0..5,999), the emo train transcripts and,
with --hkcancor, the hand-annotated HKCanCor utterances (tmp/synth_plan/hkcancor_utts.json; spoken Cantonese with human-chosen
readings), 5-24 syllables. "Rare" syllables are those used in speech (training transcripts or HKCanCor) fewer than --rare
times in the training transcripts. Sentences are accepted greedily while they add weighted coverage of rare syllables
(target 20 occurrences below 100 uses, --cover below --rare); syllables listed in --weak (analyze_synth_weak.py output, smoothed
error rate >= 5%) count 3x, and n/l, s/z/c onsets 1.5x. The rest of the quota is random. Writes sentences.json (+ _first / _rest
splits), refs.json and residual.json (rare syllables the pools could not cover enough: they need new text).
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--first", type=int, default=500)
    ap.add_argument("--rare", type=int, default=1000)
    ap.add_argument("--cover", type=int, default=10)
    ap.add_argument("--refs", type=int, default=64)
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hkcancor", default=None, help="json list of {text, phone} utterances")
    ap.add_argument("--reliable", type=float, default=0.0,
                    help="keep only sentences in which every character is used with a reading that covers at least this share of the "
                         "character's uses in the training labels (e.g. 0.85); the TTS inherits the labels' reading noise (呢 ne1/ni1/nei1) "
                         "and does not reliably say variable readings. Needs ToJyutping (--extra_path).")
    ap.add_argument("--extra_path", default=None, help="directory holding ToJyutping")
    ap.add_argument("--weak", default=None, help="weak_report.json from analyze_synth_weak.py")
    args = ap.parse_args()
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text
    from tone_merger_patterns import training_syllable_counts

    rng = random.Random(args.seed)
    counts = training_syllable_counts(args.emo_path)
    train = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"]
    d = train.select_columns(["text", "phone", "duration"])
    texts, phones, durs = d["text"], d["phone"], d["duration"]
    from jyutping import normalize_jyutping_text as norm

    pool = []                                                        # (source, text, phone, syllables)
    chinese = re.compile(r"[\u4e00-\u9fff，。？！、：；「」\s]+")
    for i in range(6000, len(texts)):
        syl = norm(phones[i]).split()
        if 5 <= len(syl) <= 24 and chinese.fullmatch(texts[i]):
            pool.append(("tts", texts[i], phones[i].strip(), syl))
    emo = load_any_dataset(args.emo_path)
    emo = (emo["train"] if hasattr(emo, "keys") else emo).select_columns(["text", "labels"])
    for tx, lb in zip(emo["text"], emo["labels"]):
        syl = norm(lb).split()
        if 5 <= len(syl) <= 24 and chinese.fullmatch(tx):
            pool.append(("emo", tx, lb.strip(), syl))
    hk_syl = Counter()
    if args.hkcancor:
        for u in json.load(open(args.hkcancor)):
            syl = u["phone"].split()
            if 5 <= len(syl) <= 24:
                pool.append(("hkcancor", u["text"], u["phone"], syl)); hk_syl.update(syl)
    if args.reliable > 0:
        if args.extra_path:
            sys.path.insert(0, args.extra_path)
        import ToJyutping as tj
        from lexical_experiment import build_lexicon

        _, r1, _, _, _ = build_lexicon(tj, args.emo_path)

        def reliable(tx, syl):
            chars = [c for c in tx if "\u4e00" <= c <= "\u9fff"]
            if len(chars) != len(syl):
                return False
            for c, s in zip(chars, syl):
                tot = sum(r1[c].values()) if c in r1 else 0
                if tot and r1[c][s] / tot < args.reliable:
                    return False
            return True

        before = len(pool)
        pool = [p for p in pool if reliable(p[1], p[3])]
        print(f"reliable-reading filter (>= {args.reliable}): kept {len(pool)} of {before} candidate sentences", flush=True)
    rng.shuffle(pool)
    used = {s for s, c in counts.items() if c > 0} | set(hk_syl)       # syllables actually used in speech
    rare = {s for s in used if counts.get(s, 0) < args.rare}
    target = {s: (20 if counts.get(s, 0) < 100 else args.cover) for s in rare}
    weight = {s: 1.0 for s in rare}
    if args.weak and os.path.exists(args.weak):
        for w in json.load(open(args.weak))["weak_syllables"]:
            if w["syllable"] in weight:
                weight[w["syllable"]] *= 3.0
    for s in weight:
        if s[:1] in ("n", "l") and not s.startswith("ng") or s[:1] in ("s", "z", "c"):
            weight[s] *= 1.5
    cov = Counter()
    chosen, taken = [], set()
    for threshold in (6.0, 3.0, 1.5, 0.75):
        for k, (src, tx, ph, syl) in enumerate(pool):
            if len(chosen) >= args.n:
                break
            if k in taken:
                continue
            gain = sum(weight[s] for s in set(syl) if s in rare and cov[s] < target[s])
            if gain >= threshold:
                chosen.append(k); taken.add(k)
                for s in syl:
                    cov[s] += 1
    for k in range(len(pool)):                                       # fill the rest at random
        if len(chosen) >= args.n:
            break
        if k not in taken:
            chosen.append(k); taken.add(k)
    rows = [{"id": f"syn{n:05d}", "src": pool[k][0], "text": pool[k][1], "phone": pool[k][2]} for n, k in enumerate(chosen)]
    residual = sorted((s for s in rare if cov[s] < target[s]), key=lambda s: cov[s])
    print("sources:", dict(Counter(r["src"] for r in rows)), f"| speech-used syllables {len(used)}, rare {len(rare)}")
    print(f"rare syllables reaching their coverage target: {len(rare) - len(residual)} of {len(rare)}; "
          f"not covered at all: {sum(1 for s in rare if cov[s] == 0)}; still short: {len(residual)}")
    in_rare = sum(1 for r in rows for s in normalize_jyutping_text(r["phone"]).split() if s in rare)
    tot = sum(len(normalize_jyutping_text(r["phone"]).split()) for r in rows)
    print(f"{len(rows)} sentences, {tot} syllables, {100 * in_rare / tot:.1f}% of them rare (training count < {args.rare})")
    # reference voices: training clips with 2.5-6 s
    longer = [i for i in range(6000, len(durs)) if 2.5 <= durs[i] <= 6.0]
    refs = rng.sample(longer, args.refs)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out_dir, "sentences.json"), "w"), ensure_ascii=False)
    json.dump(rows[: args.first], open(os.path.join(args.out_dir, "sentences_first.json"), "w"), ensure_ascii=False)
    json.dump(rows[args.first:], open(os.path.join(args.out_dir, "sentences_rest.json"), "w"), ensure_ascii=False)
    json.dump(refs, open(os.path.join(args.out_dir, "refs.json"), "w"))
    json.dump([{"syllable": s, "covered": cov[s], "target": target[s], "train_count": counts.get(s, 0), "hkcancor": hk_syl.get(s, 0)} for s in residual],
              open(os.path.join(args.out_dir, "residual.json"), "w"), ensure_ascii=False, indent=0)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
