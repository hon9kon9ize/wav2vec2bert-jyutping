"""Decode the training (or test) clips of dataset_emo_jyutping with a checkpoint, in shards (batch size 1, fp32, the decode of decode_train.py).

    python decode_emo_train.py --model_id <dir> --shard I --num_shards W --out decode_emo/decoded_emo_I.jsonl [--split train] [--limit N]

The audio is resampled to 16 kHz by `datasets` exactly as in training. One JSON line per clip: train_idx (row of the split), text
(simplified-script written Cantonese), gold (label syllables), pred. Resumable per shard.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    import numpy as np
    import torch
    from datasets import Audio

    from dataset_utils import load_any_dataset
    from error_consistency import make_predictor
    from jyutping import normalize_jyutping_text

    torch.set_num_threads(args.threads)
    data = load_any_dataset(args.emo_path)[args.split].select_columns(["audio", "text", "labels"]).cast_column("audio", Audio(sampling_rate=16_000))
    n = len(data)
    div, mod = divmod(n, args.num_shards)
    start = div * args.shard + min(args.shard, mod)
    end = start + div + (1 if args.shard < mod else 0)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["train_idx"] for line in open(args.out)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"shard {args.shard}/{args.num_shards}: rows {start}-{end} of {n} ({args.split}), {len(done)} already decoded", flush=True)
    predict = make_predictor("ours", args.model_id)
    t0, count = time.time(), 0
    with open(args.out, "a") as out:
        for i in range(start, end):
            if i in done:
                continue
            row = data[i]
            y = np.asarray(row["audio"]["array"], dtype=np.float32)
            out.write(json.dumps({"train_idx": i, "text": row["text"], "gold": normalize_jyutping_text(row["labels"]).split(), "pred": predict(y)},
                                 ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                out.flush()
                print(f"shard {args.shard}: {count} clips, {count / (time.time() - t0):.1f} clips/s", flush=True)
            if args.limit and count >= args.limit:
                return
    print(f"shard {args.shard}: done, {count} clips decoded in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
