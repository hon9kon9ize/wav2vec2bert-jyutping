"""Score single-head checkpoints on the hand-labelled wavs in tmp/labels.md (syllable edit distance, length-aware decode).

    python score_wavs.py checkpoints/run21_40000 [checkpoints/other ...] [--labels tmp/labels.md] [--quiet]
"""

import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import align, make_predictor  # noqa: E402


def main():
    os.environ.setdefault("USE_TF", "0")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--labels", default=os.path.join(REPO, "tmp", "labels.md"))
    ap.add_argument("--quiet", action="store_true", help="totals only")
    args = ap.parse_args()
    import librosa
    import numpy as np

    from jyutping import normalize_jyutping_text

    wavs = []
    for line in open(args.labels, encoding="utf-8"):
        m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|", line)
        if m and m.group(1).endswith(".wav"):
            path = os.path.join(os.path.dirname(args.labels), m.group(1))
            wavs.append((m.group(1), librosa.load(path, sr=16000, mono=True)[0].astype(np.float32), normalize_jyutping_text(m.group(2)).split()))
    n = sum(len(g) for _, _, g in wavs)
    for model in args.models:
        predict = make_predictor("ours", model)
        tot = 0
        lines = []
        for name, y, g in wavs:
            pred = predict(y)
            ops = align(g, pred)
            e = sum(op != "=" for op, *_ in ops)
            tot += e
            diffs = [f"{a}->{b}" if op == "S" else f"{a}->(del)" if op == "D" else f"(ins)->{b}" for op, _, a, b in ops if op != "="]
            lines.append(f"   [{name[:22]:22}] {e:2d}/{len(g):2d}  {diffs if diffs else 'exact'}")
        print(f"{model}: {tot}/{n} = {100 * tot / n:.1f}% syllable errors on {len(wavs)} wavs")
        if not args.quiet:
            print("\n".join(lines))


if __name__ == "__main__":
    main()
