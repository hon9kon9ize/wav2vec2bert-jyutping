"""Compare the light merger-smoothing runs (run_ms.sh): one-hot baseline versus merger-aware smoothing at several weights.

    python merger_smoothing_report.py --model ms_0=0 --model ms_005=0.005 --model ms_015=0.015 --model ms_05=0.05 --model ms_15=0.15

Needs, per model, tmp/preds_<name>.json (error_consistency.py decode), tmp/feats_<name>.json (check_tone_pitch.py build) and
tmp/lexical_<name>.txt (lexical_experiment.py). The first --model is the baseline.
"""

import argparse
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import TONE_MERGERS, align, parse  # noqa: E402

PARTNERS = {"2": ["5"], "5": ["2"], "3": ["6"], "6": ["3", "4"], "4": ["6"]}
GROUPS = {"tts": ("tts_val", "tts_test"), "emo": ("emo_test",)}


def clip_errors(c):
    e = m = 0
    for op, _, r, h in align(c["gold"], c["pred"]):
        if op != "=":
            e += 1
            if op == "S":
                po, pn, pc, pt = parse(r)
                ho, hn, hc, ht = parse(h)
                m += (po, pn, pc) == (ho, hn, hc) and pt != ht and frozenset((pt, ht)) in TONE_MERGERS
    return e, m


def auroc(pos, neg):
    """probability that a random positive scores above a random negative"""
    import numpy as np

    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=EPS")
    args = ap.parse_args()
    models = [(s.split("=")[0], s.split("=")[1]) for s in args.model]
    rng = np.random.default_rng(0)
    data, errs = {}, {}
    for name, _ in models:
        preds = json.load(open(os.path.join(REPO, "tmp", f"preds_{name}.json")))
        data[name] = {g: [c for c in preds if c["set"] in sets] for g, sets in GROUPS.items()}
        errs[name] = {g: [clip_errors(c) for c in cl] for g, cl in data[name].items()}
    base = models[0][0]

    print("=" * 112)
    print("1. Syllable errors against the labels (light runs: 6,000 clips, 1,000 steps, from the published encoder, same seed)")
    print(f"{'model':8} {'weight':>7} | {'tts errors':>11} {'SER':>6} {'vs baseline [95% CI]':>24} | {'emo errors':>11} {'SER':>6} {'vs baseline [95% CI]':>24} | merger-pair tone errors tts / emo")
    for name, eps in models:
        cells = []
        for g in GROUPS:
            e = np.array([x[0] for x in errs[name][g]]); eb = np.array([x[0] for x in errs[base][g]])
            n = sum(len(c["gold"]) for c in data[name][g])
            d = e - eb
            boots = sorted(d[rng.integers(0, len(d), len(d))].sum() for _ in range(1000))
            cells.append(f"{e.sum():>11} {100 * e.sum() / n:5.2f}% {f'{d.sum():+d} [{boots[25]:+.0f}, {boots[975]:+.0f}]':>24}")
        mt = sum(x[1] for x in errs[name]["tts"]); me = sum(x[1] for x in errs[name]["emo"])
        print(f"{name:8} {eps:>7} | " + " | ".join(cells) + f" | {mt} / {me}")

    print("\n" + "=" * 112)
    print("2. Shape of the tone posterior at the rime's peak frame, gold tone in a merger pair, rime unchanged")
    print("   partner share = the partner tone's share of the posterior over the rime's tones (T6 has two partners: summed);")
    print("   gold / between / partner = share of the pair's mass on the gold tone >= 0.9 / in between / <= 0.1")
    print(f"{'model':8} {'weight':>7} | {'median partner share T2,T3,T4,T5':>34} | {'gold':>6} {'between':>8} {'partner':>8} | NLL of the gold tone (correct / wrong tone) | tone-error detection AUROC")
    for name, eps in models:
        blob = json.load(open(os.path.join(REPO, "tmp", f"feats_{name}.json")))
        recs, clips = blob["recs"], blob["clips"]
        share, cats, nll_ok, nll_bad, conf_ok, conf_bad = [], [0, 0, 0], [], [], [], []
        for r in recs:
            if r["set"] not in ("tts_val", "tts_test", "emo_test") or parse(r["L"])[:3] != parse(r["P"])[:3]:
                continue
            tp = clips[f"{r['set']}:{r['idx']}"]["tp"][r["j"]]
            if len(tp) < 2 or r["L"][-1] not in tp:
                continue
            p = {t: math.exp(v) for t, v in tp.items()}
            tot = sum(p.values())
            p = {t: v / tot for t, v in p.items()}
            g = r["L"][-1]
            (nll_ok if r["P"] == r["L"] else nll_bad).append(-math.log(max(p[g], 1e-6)))
            (conf_ok if r["P"] == r["L"] else conf_bad).append(1 - max(p.values()))
            if g in PARTNERS and g in tp:
                ps = [p[h] for h in PARTNERS[g] if h in p]
                if ps and g in "2345":
                    share.append(sum(ps))
                for h in PARTNERS[g]:
                    if h in p and g in "2345":
                        q = p[g] / (p[g] + p[h])
                        cats[0 if q >= 0.9 else 2 if q <= 0.1 else 1] += 1
        tot = max(sum(cats), 1)
        print(f"{name:8} {eps:>7} | {100 * np.median(share):>33.1f}% | {100 * cats[0] / tot:5.1f}% {100 * cats[1] / tot:7.1f}% {100 * cats[2] / tot:7.1f}% | "
              f"{np.mean(nll_ok):6.3f} / {np.mean(nll_bad):6.2f}   (mean over all {np.mean(nll_ok + nll_bad):.3f}) | "
              f"{auroc(np.array(conf_bad), np.array(conf_ok)):.3f}")

    print("\n" + "=" * 112)
    print("3. Lexical decoder (character LM + the model's tone posterior); beta tuned on tts validation; errors model alone -> decoder")
    for name, eps in models:
        txt = open(os.path.join(REPO, "tmp", f"lexical_{name}.txt")).read()
        m = re.search(r"beta tuned on tts_val: ([0-9.]+)", txt)
        if not m:
            print(f"   {name}: (no result)")
            continue
        beta = m.group(1)
        row = re.search(rf"^\s*{re.escape(beta)}\s+(.*)$", txt.split("beta  ")[-1], re.M)
        model_row = re.search(r"^ model\s+(.*)$", txt, re.M)
        print(f"   {name:8} weight {eps:>6}  beta {beta:>4}  model alone: {model_row.group(1).strip()}")
        print(f"   {'':8} {'':13}            decoder:     {row.group(1).strip()}")


if __name__ == "__main__":
    main()
