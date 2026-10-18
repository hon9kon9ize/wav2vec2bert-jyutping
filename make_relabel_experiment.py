"""Training subset and corrected labels for the small relabel experiment (only the `always` rules of relabel_rules.py).

    python make_relabel_experiment.py --tojyutping_path <dir>

The subset is drawn from the tts training split WITHOUT the 20k decoded clips (the listening samples come from those, so the
verified clips stay unseen): every clip with a label that a rule would change, clips whose label already has the rule's
target reading (so the contrast stays in the data), and random other clips. Writes to tmp/label_mining/relabel_exp/:
  train_indices.json   indices into the fixed train split (feed to `compare_heads.py --train_indices`)
  override.json        {index: corrected toned jyutping} for the clips a rule changes (feed to `--phone_override`)
Run compare_heads.py twice with the same indices and seed, with and without the override.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from lexical_experiment import char_syllable_pairs  # noqa: E402
from mine_label_disagreements import OUT_DIR  # noqa: E402
from relabel_rules import RULES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tojyutping_path", default=None)
    ap.add_argument("--label_dir", default=OUT_DIR)
    ap.add_argument("--out_dir", default=os.path.join(OUT_DIR, "relabel_exp"))
    ap.add_argument("--n_correct", type=int, default=3000, help="clips whose label already has a rule's target reading")
    ap.add_argument("--n_random", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text

    rules = [(c, f, t) for c, f, t, w in RULES if w == "always"]
    exclude = {json.loads(line)["train_idx"] for line in open(os.path.join(args.label_dir, "decoded.jsonl"))}
    train = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select_columns(["text", "phone"])
    override, correct, other, changed = {}, [], [], Counter()
    for idx, (text, phone) in enumerate(zip(train["text"], train["phone"])):
        if idx in exclude:
            continue
        syllables = normalize_jyutping_text(phone).split()
        pairs = char_syllable_pairs(tj, text, syllables)
        if pairs is None:
            continue
        fixed, has_target = list(syllables), False
        for i, (ch, s) in enumerate(pairs):
            for char, frm, to in rules:
                if ch == char and s == frm:
                    fixed[i] = to
                    changed[f"{char} {frm}->{to}"] += 1
                elif ch == char and s == to:
                    has_target = True
        if fixed != syllables:
            override[idx] = " ".join(fixed)
        elif has_target:
            correct.append(idx)
        else:
            other.append(idx)
    rng = random.Random(args.seed)
    picked = sorted(set(override) | set(rng.sample(correct, min(args.n_correct, len(correct)))) | set(rng.sample(other, min(args.n_random, len(other)))))
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(picked, open(os.path.join(args.out_dir, "train_indices.json"), "w"))
    json.dump({str(k): v for k, v in override.items() if k in set(picked)}, open(os.path.join(args.out_dir, "override.json"), "w"), ensure_ascii=False)
    print(f"{len(exclude)} clips excluded (the 20k listening sample); {len(train) - len(exclude)} candidates")
    print(f"changed labels: {dict(changed)} in {len(override)} clips")
    print(f"training subset: {len(picked)} clips = {len(override)} with a changed label + {min(args.n_correct, len(correct))} with a target reading "
          f"+ {min(args.n_random, len(other))} other -> {args.out_dir}")


if __name__ == "__main__":
    main()
