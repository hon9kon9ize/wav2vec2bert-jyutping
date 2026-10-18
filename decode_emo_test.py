"""SER of a checkpoint on the 1,035-clip test split of dataset_emo_jyutping (batch size 1, fp32, length-aware decode).

    python decode_emo_test.py --model_id <dir> --emo_path <dataset_emo_jyutping> --out out/emo_test_<name>.json
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--emo_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import numpy as np
    from datasets import Audio

    from dataset_utils import load_any_dataset
    from error_consistency import make_predictor
    from jyutping import normalize_jyutping_text

    def edit_distance(ref, hyp):
        prev = list(range(len(hyp) + 1))
        for i, r in enumerate(ref, 1):
            cur = [i]
            for j, h in enumerate(hyp, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
            prev = cur
        return prev[-1]

    test = load_any_dataset(args.emo_path)["test"].select_columns(["audio", "labels"]).cast_column("audio", Audio(sampling_rate=16_000))
    predict = make_predictor("ours", args.model_id)
    errors = total = 0
    for row in test:
        gold = normalize_jyutping_text(row["labels"]).split()
        errors += edit_distance(gold, predict(np.asarray(row["audio"]["array"], dtype=np.float32)))
        total += len(gold)
    json.dump({"model": args.model_id, "clips": len(test), "syllables": total, "errors": errors, "ser": errors / total}, open(args.out, "w"))
    print(f"emo test: {len(test)} clips, {errors} / {total} syllables wrong, SER {errors / total:.4%}")


if __name__ == "__main__":
    main()
