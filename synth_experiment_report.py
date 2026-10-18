"""Compare the light runs trained with synthetic data / augmentation (run_se.sh) against the real-only baseline.

    python synth_experiment_report.py --model ms_0 --model se_S --model se_SA --model se_RA --model se_R

Reads tmp/preds_<name>.json (real held-out decode), tmp/wavs_<name>.txt and tmp/probe_{q150,rare}_<name>.txt. First --model is the baseline.
"""
import argparse, json, os, re, sys
from collections import Counter
REPO = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, REPO)
from error_consistency import align, parse  # noqa: E402

GROUPS = {"tts": ("tts_val", "tts_test"), "emo": ("emo_test",)}


def clip_err(c):
    return sum(op != "=" for op, *_ in align(c["gold"], c["pred"]))


def main():
    import numpy as np
    from tone_merger_patterns import training_syllable_counts
    ap = argparse.ArgumentParser(); ap.add_argument("--model", action="append", required=True); args = ap.parse_args()
    names = args.model; rng = np.random.default_rng(0)
    cnt = training_syllable_counts("/home/joseph/data/dataset_emo_jyutping")
    data = {n: json.load(open(f"tmp/preds_{n}.json")) for n in names}
    base = names[0]
    print(f"{'model':8} | {'tts errors':>10} {'SER':>6} {'vs base [95% CI]':>18} | {'emo errors':>10} {'SER':>6} {'vs base [95% CI]':>18} | {'wavs /79':>8} | {'synth held-out':>14} | {'synth rare-syll.':>16} | real rare<1000: errors/syll")
    for n in names:
        cells = []
        for g, sets in GROUPS.items():
            cl = [c for c in data[n] if c["set"] in sets]; cb = [c for c in data[base] if c["set"] in sets]
            e = np.array([clip_err(c) for c in cl]); eb = np.array([clip_err(c) for c in cb]); k = sum(len(c["gold"]) for c in cl)
            d = e - eb; boots = sorted(d[rng.integers(0, len(d), len(d))].sum() for _ in range(1000))
            cells.append(f"{e.sum():>10} {100 * e.sum() / k:5.2f}% {f'{d.sum():+d} [{boots[25]:+.0f},{boots[975]:+.0f}]':>18}")
        def grab(path, pat):
            try: return re.search(pat, open(path).read()).group(1)
            except Exception: return "-"
        w = grab(f"tmp/wavs_{n}.txt", r": (\d+)/79")
        q = grab(f"tmp/probe_q150_{n}.txt", r"SER ([0-9.]+)%"); r = grab(f"tmp/probe_rare_{n}.txt", r"SER ([0-9.]+)%")
        ne = nn = 0
        for c in data[n]:
            for op, i, a, b in align(c["gold"], c["pred"]):
                if op in ("=", "S", "D") and cnt.get(a, 0) < 1000:
                    nn += 1; ne += op != "="
        print(f"{n:8} | " + " | ".join(cells) + f" | {w:>8} | {q + '%':>14} | {r + '%':>16} | {ne}/{nn} = {100 * ne / nn:.2f}%")
if __name__ == "__main__":
    main()
