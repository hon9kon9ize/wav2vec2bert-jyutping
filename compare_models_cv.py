"""Compare two jyutping models on real recorded speech: the Common Voice clips of the Qwen project's label audit.

    python compare_models_cv.py --a run21=<pred_local.jsonl> --b retrain_top8=<pred.jsonl> [--out_dir tmp/label_mining/cv_eval]

Predictions come from that project's `run_jyutping.py` (same decode, bf16). The reference is ToJyutping of the (audited) transcript and the
scoring is that project's `compare_jyutping.py` (imported, not copied): strict = every syllable that differs, reading = another attested
reading of the character is accepted, hard = documented merger pairs are accepted too. None of these clips was ever in the training data of
either model. Caveat: the Qwen project rewrote 1,334 clause-final particles (呀 啊 吖 啦 喇 嘞 囉 咯 囖) in these transcripts from the syllable
run21 itself heard, which biases the references of those characters toward run21; the scores are therefore given with those characters
dropped (default) and with them kept. The paired clip-level bootstrap gives the 95% interval of the difference (b minus a).
Writes summary.json and disagreements.tsv (every syllable where the two models differ, for listening) to --out_dir.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

QWEN = "/home/joseph/projects/qwen3asr-analysis/training_data/label_audit"
sys.path.insert(0, QWEN)
import compare_jyutping as cj  # noqa: E402

PARTICLES = set("啊呀吖啦喇嘞囉咯囖")
REPO = os.path.dirname(os.path.abspath(__file__))


def load_preds(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def score(row, pred, drop_particles):
    """(reference syllables counted, strict errors, reading errors, hard errors, per-character hard errors) of one clip, as in
    compare_jyutping.score_clip: strict = hard + merger + other readings, reading = hard + merger, hard = substitutions that are not a
    merger pair + deleted + inserted syllables"""
    ref, chars = row["ref"], row["chars"]
    ops = cj.align(ref, pred["pred"], row["cands"])
    n = strict = reading = hard = 0
    per_char = Counter()
    for op, i, j in ops:
        if op == "I":
            strict += 1; reading += 1; hard += 1
            continue
        char = chars[i] if i < len(chars) else "?"
        if drop_particles and char in PARTICLES:
            continue
        n += 1
        if op == "M":
            continue
        strict += 1
        if op == "R":
            continue
        reading += 1
        if op == "D" or cj.classify(ref[i], pred["pred"][j]) != "merger":
            hard += 1
            per_char[char] += 1
    return n, strict, reading, hard, per_char


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, metavar="NAME=PRED.jsonl")
    ap.add_argument("--b", required=True, metavar="NAME=PRED.jsonl")
    ap.add_argument("--manifest", default=os.path.join(QWEN, "manifest.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "tmp", "label_mining", "cv_eval"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    (name_a, path_a), (name_b, path_b) = args.a.split("=", 1), args.b.split("=", 1)
    pa, pb = load_preds(path_a), load_preds(path_b)
    rows = {}
    for line in open(args.manifest):
        r = json.loads(line)
        if r["src"] in ("cv", "cvo") and not r.get("has_latin") and r["id"] in pa and r["id"] in pb and r["ref"]:
            rows[r["id"]] = r
    print(f"{len(rows)} Common Voice clips scored (no Latin text, predicted by both models)")

    summary = {}
    for drop in (True, False):
        label = "particles dropped" if drop else "particles kept"
        per_clip = {}
        for cid, r in rows.items():
            per_clip[cid] = {name_a: score(r, pa[cid], drop), name_b: score(r, pb[cid], drop)}
        print(f"\n=== {label}")
        print(f"{'source':8} {'clips':>6} {'syllables':>10} | {'level':8} {name_a:>12} {name_b:>12} {'diff (b-a)':>11}  95% CI of the difference")
        for src in ("cv", "cvo", "all"):
            ids = [c for c, r in rows.items() if src == "all" or r["src"] == src]
            n = sum(per_clip[c][name_a][0] for c in ids)
            for k, level in ((1, "strict"), (2, "reading"), (3, "hard")):
                ea = sum(per_clip[c][name_a][k] for c in ids)
                eb = sum(per_clip[c][name_b][k] for c in ids)
                rng = random.Random(0)
                diffs = []
                deltas = [(per_clip[c][name_b][k] - per_clip[c][name_a][k], per_clip[c][name_a][0]) for c in ids]
                for _ in range(args.bootstrap):
                    s = [deltas[rng.randrange(len(deltas))] for _ in deltas]
                    diffs.append(sum(d for d, _ in s) / max(sum(m for _, m in s), 1))
                diffs.sort()
                lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
                print(f"{src:8} {len(ids):6d} {n:10d} | {level:8} {ea / n:12.2%} {eb / n:12.2%} {(eb - ea) / n:+11.2%}  {lo:+.2%} to {hi:+.2%}")
                summary[f"{label}/{src}/{level}"] = {name_a: ea / n, name_b: eb / n, "diff": (eb - ea) / n, "ci": [lo, hi], "syllables": n}
        if drop:
            ca, cb, occ = Counter(), Counter(), Counter()
            for cid, r in rows.items():
                ca.update(per_clip[cid][name_a][4]); cb.update(per_clip[cid][name_b][4])
                occ.update(ch for ch in r["chars"] if ch not in PARTICLES)
            moves = sorted(((cb[c] - ca[c], c) for c in set(ca) | set(cb) if occ[c] >= 30), key=lambda x: x[0])
            fmt = lambda xs: "  ".join(f"{c}:{ca[c]}->{cb[c]} (of {occ[c]})" for _, c in xs)
            print(f"\ncharacters (>= 30 occurrences) whose hard errors drop most from {name_a} to {name_b}:\n  " + fmt(moves[:12]))
            print(f"characters whose hard errors rise most:\n  " + fmt(moves[-12:][::-1]))
            summary["char_moves"] = {"down": [c for _, c in moves[:12]], "up": [c for _, c in moves[-12:]]}

    os.makedirs(args.out_dir, exist_ok=True)
    n_diff = 0
    with open(os.path.join(args.out_dir, "disagreements.tsv"), "w") as f:
        f.write("clip\tsrc\tpos\tchar\treference\t%s\t%s\ttext\n" % (name_a, name_b))
        clips_diff = 0
        for cid, r in rows.items():
            if pa[cid]["pred"] == pb[cid]["pred"]:
                continue
            clips_diff += 1
            a, b = pa[cid]["pred"], pb[cid]["pred"]
            for op, i, j in cj.align(a, b, [[x] for x in a]):
                if op != "S":
                    continue
                # position in the reference: the syllable of a that the alignment of a to the reference matched (approximate by index)
                pos = i
                ch = r["chars"][pos] if pos < len(r["chars"]) and len(a) == len(r["ref"]) else "?"
                ref = r["ref"][pos] if pos < len(r["ref"]) and len(a) == len(r["ref"]) else "?"
                f.write(f"{cid}\t{r['src']}\t{i}\t{ch}\t{ref}\t{a[i]}\t{b[j]}\t{r['text']}\n")
                n_diff += 1
    print(f"\nthe two models give different syllables in {clips_diff} of {len(rows)} clips ({n_diff} syllable substitutions) -> {args.out_dir}/disagreements.tsv")
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
