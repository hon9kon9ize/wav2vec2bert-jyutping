"""Apply the label corrections of relabel_rules.RULES to the tts training labels (nothing in a dataset is overwritten).

    python apply_relabel_rules.py --decoded <decoded_train.jsonl> --tojyutping_path <dir> [--out_dir tmp/label_mining/relabel_full]

--decoded is a decode of the training clips (decode_train.py; the 20k decode of mine_label_disagreements.py has the same fields).
A rule changes a label syllable when its character has the rule's old reading and, for `model` rules, the model's syllable at
that position is the corrected reading. Writes to --out_dir:
  corrected_labels.json   {train-split index: corrected toned jyutping} for every clip a rule changes
                          (feed to `compare_heads.py --phone_override` or a training script)
  changes.jsonl           one row per changed syllable (train_idx, position, character, old, new, rule, model, text) for spot checks
"""

import argparse
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import align  # noqa: E402
from lexical_experiment import char_syllable_pairs  # noqa: E402
from mine_label_disagreements import OUT_DIR  # noqa: E402
from relabel_rules import RULES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoded", required=True)
    ap.add_argument("--tojyutping_path", default=None)
    ap.add_argument("--out_dir", default=os.path.join(OUT_DIR, "relabel_full"))
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    by_label = {}
    for char, frm, to, when in RULES:
        by_label.setdefault((char, frm), []).append((to, when))
    seen, changed, clips, skipped, corrected, rows = Counter(), Counter(), 0, 0, {}, []
    for line in open(args.decoded):
        clip = json.loads(line)
        clips += 1
        pairs = char_syllable_pairs(tj, clip["text"], clip["gold"])
        if pairs is None:
            skipped += 1
            continue
        fixed, model_at = list(clip["gold"]), None
        for i, (char, label) in enumerate(pairs):
            for to, when in by_label.get((char, label), ()):
                seen[(char, label, to)] += 1
                if model_at is None:
                    model_at = {k: (g if op == "=" else p) for op, k, g, p in align(clip["gold"], clip["pred"]) if k is not None}
                if when == "model" and model_at.get(i) != to:
                    continue
                fixed[i] = to
                changed[(char, label, to, when)] += 1
                rows.append({"train_idx": clip["train_idx"], "pos": i, "char": char, "old": label, "new": to, "rule": when,
                             "model": model_at.get(i), "text": clip["text"]})
        if fixed != clip["gold"]:
            corrected[str(clip["train_idx"])] = " ".join(fixed)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(corrected, open(os.path.join(args.out_dir, "corrected_labels.json"), "w"), ensure_ascii=False)
    with open(os.path.join(args.out_dir, "changes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{clips} clips read, {skipped} skipped (syllable count differs from ToJyutping(text)), {len(corrected)} clips changed, {len(rows)} syllables changed")
    print(f"{'rule':16} {'when':7} {'old label seen':>15} {'changed':>8}")
    for char, frm, to, when in RULES:
        print(f"{char + ' ' + frm + '->' + to:16} {when:7} {seen[(char, frm, to)]:15d} {changed[(char, frm, to, when)]:8d}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
