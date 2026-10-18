"""Decode every tts20250516 training clip with a checkpoint, in contiguous shards of the stored (unshuffled) dataset.

    python decode_train.py --model_id <dir> --shard I --num_shards W --out out/decoded_train_I.jsonl [--limit N]

The fixed train split is a shuffle (seed 42) of the whole dataset, and reading it in order is random access on the stored
files (about 5 clips/s on genkai's Lustre). So the stored rows are read in order instead, and each row's index in the train
split is recovered from the same permutation (`np.random.default_rng(42).permutation`, checked against `Dataset.shuffle`).
The decode is the one used for the 20k sample of mine_label_disagreements.py: batch size 1, fp32, length-aware, so "the model
says X" means the same thing. One JSON line per clip: train_idx, text, gold (label syllables), pred. Resumable per shard.
"""

import argparse
import io
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

VALIDATION_TEST = 1000  # the first 500 + 500 positions of the shuffled dataset are the test and validation splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many decoded clips (testing)")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio

    from dataset_utils import DEFAULT_DATASET, flatten_dataset, load_any_dataset
    from error_consistency import make_predictor
    from jyutping import normalize_jyutping_text

    torch.set_num_threads(args.threads)
    flat = flatten_dataset(load_any_dataset(DEFAULT_DATASET))
    n = len(flat)
    perm = np.random.default_rng(42).permutation(n)
    position = np.empty(n, dtype=np.int64)  # shuffled position of every stored row
    position[perm] = np.arange(n)
    text_phone = flat.select_columns(["text", "phone"])
    shuffled = text_phone.shuffle(seed=42)
    for p in (0, 1, n // 3, n // 2, n - 1):
        assert shuffled[p] == text_phone[int(perm[p])], f"permutation differs from Dataset.shuffle at {p}"

    div, mod = divmod(n, args.num_shards)
    start = div * args.shard + min(args.shard, mod)
    end = start + div + (1 if args.shard < mod else 0)
    part = flat.select(range(start, end)).cast_column("audio", Audio(decode=False))
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["train_idx"] for line in open(args.out)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"shard {args.shard}/{args.num_shards}: stored rows {start}-{end} of {n}, {len(done)} already decoded", flush=True)

    predict = make_predictor("ours", args.model_id)
    t0, count = time.time(), 0
    with open(args.out, "a") as out:
        for lo in range(0, end - start, 64):
            chunk = part[lo: lo + 64]
            for j in range(len(chunk["text"])):
                p = int(position[start + lo + j])
                if p < VALIDATION_TEST or (p - VALIDATION_TEST) in done:
                    continue
                audio = chunk["audio"][j]
                y, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if rate != 16000:
                    import librosa

                    y = librosa.resample(y, orig_sr=rate, target_sr=16000)
                out.write(json.dumps({"train_idx": p - VALIDATION_TEST, "text": chunk["text"][j],
                                      "gold": normalize_jyutping_text(chunk["phone"][j]).split(), "pred": predict(y.astype(np.float32))},
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
