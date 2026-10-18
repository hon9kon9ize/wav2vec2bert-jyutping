"""Where is the model weak? A systematic map from synthetic clips whose label is exact by construction, used to decide what
else to synthesize.

    python analyze_synth_weak.py --synth_dir tmp/synth_plan/out_first [--synth_dir tmp/synth_plan/out_rest] \\
        [--model_id checkpoints/run21_40000] [--out tmp/synth_plan/weak_report.json]

Decodes every clip (cached in <out>.preds.json), then reports the error rate by training frequency of the gold syllable, by
part (onset / nucleus / coda / tone) and token, the weakest syllables and confusion pairs, position in the sentence and voice
type, and writes weak_report.json (weakest syllables with a smoothed error rate and their usual wrong answers) for the next
round of sentence selection.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import ONSET_MERGERS, TONE_MERGERS, align, make_predictor, parse  # noqa: E402


def main():
    os.environ.setdefault("USE_TF", "0")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth_dir", action="append", required=True)
    ap.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    ap.add_argument("--out", default=os.path.join(REPO, "tmp", "synth_plan", "weak_report.json"))
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--min_n", type=int, default=6, help="minimum occurrences for a syllable to be ranked")
    args = ap.parse_args()
    import librosa
    import numpy as np

    from jyutping import normalize_jyutping_text
    from tone_merger_patterns import training_syllable_counts

    items = []
    for d in args.synth_dir:
        for m in json.load(open(os.path.join(d, "manifest.json"))):
            items.append(dict(m, path=os.path.join(d, m["file"])))
    cache = args.out + ".preds.json"
    preds = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [it for it in items if it["file"] not in preds]
    if todo:
        predict = make_predictor("ours", args.model_id)
        for n, it in enumerate(todo):
            y = librosa.load(it["path"], sr=16000, mono=True)[0].astype(np.float32)
            preds[it["file"]] = predict(y)
            if n % 500 == 0:
                print(f"decoded {n}/{len(todo)}", flush=True)
        json.dump(preds, open(cache, "w"))
    counts = training_syllable_counts(args.emo_path)

    syl_n, syl_e, syl_wrong = Counter(), Counter(), defaultdict(Counter)
    part_n, part_e = {k: Counter() for k in ("onset", "nuc", "coda", "tone")}, {k: Counter() for k in ("onset", "nuc", "coda", "tone")}
    pair_c = {k: Counter() for k in ("onset", "nuc", "coda", "tone")}
    freq_n, freq_e = Counter(), Counter()
    pos_n, pos_e = Counter(), Counter()
    voice_n, voice_e, clips_exact = Counter(), Counter(), Counter()
    merger_only = tot_err = tot_syl = 0
    BUCKETS = [(0, 100), (100, 1000), (1000, 10000), (10000, 10 ** 9)]
    for it in items:
        gold, pred = normalize_jyutping_text(it["phone"]).split(), preds[it["file"]]
        vt = "own wav voice" if not it["ref"].startswith("ref") else "training-corpus voice"
        L, e_clip = len(gold), 0
        for op, i, r, h in align(gold, pred):
            if op == "I":
                tot_err += 1; e_clip += 1; continue
            c = counts.get(r, 0)
            b = next(k for k, (lo, hi) in enumerate(BUCKETS) if lo <= c < hi)
            pos = "first" if i == 0 else "last" if i == L - 1 else "middle"
            wrong = op != "="
            tot_syl += 1; syl_n[r] += 1; freq_n[b] += 1; pos_n[pos] += 1; voice_n[vt] += 1
            po, pn, pc, pt = parse(r)
            g = {"onset": po or "-", "nuc": pn, "coda": pc or "-", "tone": pt}
            for k in g:
                part_n[k][g[k]] += 1
            if wrong:
                tot_err += 1; e_clip += 1; syl_e[r] += 1; freq_e[b] += 1; pos_e[pos] += 1; voice_e[vt] += 1
                if op == "S":
                    syl_wrong[r][h] += 1
                    ho, hn, hc, ht = parse(h)
                    p2 = {"onset": ho or "-", "nuc": hn, "coda": hc or "-", "tone": ht}
                    diff = [k for k in g if g[k] != p2[k]]
                    for k in diff:
                        part_e[k][g[k]] += 1; pair_c[k][(g[k], p2[k])] += 1
                    ok = (not {"nuc", "coda"} & set(diff)) and ("onset" not in diff or frozenset((po or "-", ho or "-")) in ONSET_MERGERS) \
                        and ("tone" not in diff or frozenset((pt, ht)) in TONE_MERGERS)
                    merger_only += ok
                else:
                    syl_wrong[r]["(deleted)"] += 1
        clips_exact[vt] += e_clip == 0
    print(f"{len(items)} synthetic clips, {tot_syl} gold syllables, SER {100 * tot_err / tot_syl:.2f}% ({tot_err} errors); "
          f"errors that are merger pairs only: {merger_only} ({100 * merger_only / max(tot_err, 1):.0f}%)")
    for vt in voice_n:
        print(f"   {vt:24} syllables {voice_n[vt]:6}  SER {100 * voice_e[vt] / voice_n[vt]:.2f}%")
    print("\n1. Error rate by how often the gold syllable occurs in the real training transcripts")
    for k, (lo, hi) in enumerate(BUCKETS):
        print(f"   {lo:>6}-{hi if hi < 10 ** 9 else 'inf':<6} syllables {freq_n[k]:6}  errors {freq_e[k]:4}  rate {100 * freq_e[k] / max(freq_n[k], 1):5.2f}%")
    print("\n2. Position in the sentence: " + "   ".join(f"{p}: {100 * pos_e[p] / pos_n[p]:.2f}% (n={pos_n[p]})" for p in ("first", "middle", "last")))
    print("\n3. Weakest tokens by part (gold token: errors / n, error rate; n >= 100) and the usual replacements")
    for k in ("onset", "nuc", "coda", "tone"):
        rows = [(t, part_e[k][t], part_n[k][t]) for t in part_n[k] if part_n[k][t] >= 100 and part_e[k][t] >= 3]
        rows.sort(key=lambda r: -r[1] / r[2])
        out = []
        for t, e, n in rows[:6]:
            tops = ", ".join(f"{b}:{c}" for (a, b), c in pair_c[k].most_common() if a == t)[:40]
            out.append(f"{t} {e}/{n} ({100 * e / n:.1f}%) -> {tops.split(', ')[0] if tops else ''}")
        print(f"   {k:6} " + "; ".join(out))
    print("\n4. Most common confusions (gold -> model, count)")
    for k in ("tone", "onset", "nuc", "coda"):
        print(f"   {k:6} " + ", ".join(f"{a}>{b}:{c}" for (a, b), c in pair_c[k].most_common(8)))
    # smoothed per-syllable error rate (Beta prior from the overall rate)
    prior = tot_err / tot_syl
    strength = 20.0
    rank = []
    for s, n in syl_n.items():
        if n >= args.min_n:
            e = syl_e[s]
            rank.append((s, n, e, (e + prior * strength) / (n + strength)))
    rank.sort(key=lambda r: -r[3])
    print(f"\n5. Weakest syllables (n >= {args.min_n}; smoothed error rate; usual wrong answers; training count)")
    for s, n, e, sm in rank[:30]:
        wr = ", ".join(f"{h}:{c}" for h, c in syl_wrong[s].most_common(3))
        print(f"   {s:8} {e:3}/{n:3} = {100 * e / n:5.1f}% (smoothed {100 * sm:4.1f}%)  -> {wr:34} train {counts.get(s, 0)}")
    n_bad = sum(1 for r in rank if r[3] >= 0.05)
    print(f"\n   syllables ranked: {len(rank)}; smoothed error rate >= 5%: {n_bad}; >= 10%: {sum(1 for r in rank if r[3] >= 0.10)}")
    rep = {"overall_ser": tot_err / tot_syl, "weak_syllables": [
        {"syllable": s, "n": n, "errors": e, "smoothed_rate": sm, "wrong": dict(syl_wrong[s].most_common(4)), "train_count": counts.get(s, 0)}
        for s, n, e, sm in rank if sm >= 0.05],
        "top_confusions": {k: [[a, b, c] for (a, b), c in pair_c[k].most_common(40)] for k in pair_c}}
    json.dump(rep, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
