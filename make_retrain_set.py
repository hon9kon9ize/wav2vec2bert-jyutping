"""Training subset for the continued fine-tune on corrected labels (an enriched subset: the feature cache of all ~280k clips would need ~80 GB).

    python make_retrain_set.py --tojyutping_path <dir> [--corrected tmp/label_mining/relabel_full_v2/corrected_labels.json]

Drawn from the tts training split without the clips the listener judged (the 20k decode sample and every clip of the listening page, so
they stay a clean check): all clips with a corrected label (these are repeated in training), clips whose labels already carry a rule's
target reading (so the contrast stays in the data), and random other clips; plus random emo training clips as replay. Writes to --out_dir:
  train_indices.json, repeat_indices.json, override.json (tts train-split indices), emo_indices.json (rows of the emo train split)
"""

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from lexical_experiment import char_syllable_pairs  # noqa: E402
from mine_label_disagreements import OUT_DIR  # noqa: E402
from relabel_rules import RULES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tojyutping_path", default=None)
    ap.add_argument("--corrected", default=os.path.join(OUT_DIR, "relabel_full_v2", "corrected_labels.json"))
    ap.add_argument("--decoded_train", default=os.path.join(OUT_DIR, "decode_full", "decoded_train.jsonl"))
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--n_contrast", type=int, default=15000)
    ap.add_argument("--n_random", type=int, default=50000)
    ap.add_argument("--n_emo", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(OUT_DIR, "retrain"))
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    from dataset_utils import load_any_dataset

    judged = {json.loads(l)["train_idx"] for l in open(os.path.join(OUT_DIR, "decoded.jsonl"))}
    judged |= {s["clip"] for g in json.load(open(os.path.join(OUT_DIR, "tone", "samples.json")))["groups"] for s in g["samples"]}
    corrected = {int(k): v for k, v in json.load(open(args.corrected)).items()}
    targets = {(char, to) for char, _, to, _ in RULES}
    changed, contrast, other = [], [], []
    for line in open(args.decoded_train):
        clip = json.loads(line)
        k = clip["train_idx"]
        if k in judged:
            continue
        if k in corrected:
            changed.append(k)
            continue
        pairs = char_syllable_pairs(tj, clip["text"], clip["gold"])
        (contrast if pairs and any(pair in targets for pair in pairs) else other).append(k)
    rng = random.Random(args.seed)
    picked_contrast = rng.sample(contrast, min(args.n_contrast, len(contrast)))
    picked_other = rng.sample(other, min(args.n_random, len(other)))
    train = sorted(set(changed) | set(picked_contrast) | set(picked_other))
    emo_rows = len(load_any_dataset(args.emo_path)["train"])
    emo = sorted(rng.sample(range(emo_rows), min(args.n_emo, emo_rows)))
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(train, open(os.path.join(args.out_dir, "train_indices.json"), "w"))
    json.dump(sorted(changed), open(os.path.join(args.out_dir, "repeat_indices.json"), "w"))
    json.dump({str(k): corrected[k] for k in changed}, open(os.path.join(args.out_dir, "override.json"), "w"), ensure_ascii=False)
    json.dump(emo, open(os.path.join(args.out_dir, "emo_indices.json"), "w"))
    print(f"{len(judged)} judged clips left out; candidates: {len(changed)} with a corrected label, {len(contrast)} with a target reading, {len(other)} other")
    print(f"training subset: {len(train)} tts clips = {len(changed)} corrected + {len(picked_contrast)} contrast + {len(picked_other)} random, plus {len(emo)} emo clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
