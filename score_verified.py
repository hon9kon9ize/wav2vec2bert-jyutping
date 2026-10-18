"""Score models on the clips the listener has judged (tone_listening.py verdicts): does the model give the reading that was heard?

    python score_verified.py run21=checkpoints/run21_40000 A=checkpoints/relabel_exp/A_orig/final B=checkpoints/relabel_exp/B_fixed/final

For every judged syllable the truth is the reading the listener chose (the label's, or the model's / the agreement check's
alternative). A model is right when its syllable at that position equals the truth. Items are split by whether the FIRST model on
the command line was right on them: "fixed" are its errors (label right, or label and model both wrong), "kept" its right answers
(a regression check). The clips are in the 20k decode sample, which the relabel experiment leaves out of training.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import align, make_predictor  # noqa: E402
from mine_label_disagreements import OUT_DIR  # noqa: E402
from tone_listening import TONE_DIR  # noqa: E402

RULE_CHARS = "度自在"


def judged_items(tone_dir, label_dir, include_fresh=False, decoded_path=None):
    data = json.load(open(os.path.join(tone_dir, "samples.json")))
    decoded = {c["train_idx"]: c for c in map(json.loads, open(decoded_path or os.path.join(label_dir, "decoded.jsonl")))}
    verdicts = {}
    for line in open(os.path.join(tone_dir, "verdicts.jsonl")):
        r = json.loads(line)
        verdicts[r["id"]] = r["verdict"]
    items = []
    for g in data["groups"]:
        if g.get("kind") == "audit" or (g.get("kind") == "fresh" and not include_fresh):  # these clips are in the first relabel experiment's training subset
            continue
        for s in g["samples"]:
            v = verdicts.get(s["id"])
            if v not in ("A", "B"):
                continue
            label_heard = (v == "A") == s["a_is_label"]
            items.append({"id": s["id"], "clip": s["clip"], "pos": int(s["id"].split("_")[1]), "char": s["char"],
                          "truth": s["gold"] if label_heard else s["pred"], "label": s["gold"], "agree_check": g.get("kind") == "agree",
                          "gold": decoded[s["clip"]]["gold"]})
    return items


def read_wav(path):
    import wave

    import numpy as np

    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def syllable_at(gold, pred, pos):
    for op, i, _, p in align(gold, pred):
        if i == pos:
            return gold[pos] if op == "=" else p if op == "S" else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="name=model_dir; the first one defines fixed / kept")
    ap.add_argument("--tone_dir", default=TONE_DIR)
    ap.add_argument("--label_dir", default=OUT_DIR)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "relabel_exp", "verified_scores.json"))
    ap.add_argument("--decoded", default=None,
                    help="decode that holds the label syllables of every judged clip (default: label_dir/decoded.jsonl, the 20k sample; "
                         "--include_fresh needs the full decode, decode_full/decoded_train.jsonl)")
    ap.add_argument("--include_fresh", action="store_true",
                    help="also score the fresh rule-check clips (only for a model that was not trained on them)")
    args = ap.parse_args()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    items = judged_items(args.tone_dir, args.label_dir, args.include_fresh, args.decoded)
    audio = {c: read_wav(os.path.join(args.tone_dir, "wav", f"{c}.wav")) for c in {it["clip"] for it in items}}
    answers = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        if not os.path.exists(os.path.join(path, "vocab.json")):
            import shutil

            shutil.copy(os.path.join(REPO, "vocab_nucleus_tone.json"), os.path.join(path, "vocab.json"))
        predict = make_predictor("ours", path)
        preds = {c: predict(y) for c, y in audio.items()}
        answers[name] = {it["id"]: syllable_at(it["gold"], preds[it["clip"]], it["pos"]) for it in items}
        print(f"{name}: decoded {len(preds)} clips", flush=True)

    first = list(answers)[0]
    for it in items:
        it["base_right"] = answers[first][it["id"]] == it["truth"]
    groups = {
        "all judged": lambda it: True,
        f"fixed (errors of {first})": lambda it: not it["base_right"],
        "  fixed: 度/自/在": lambda it: not it["base_right"] and it["char"] in RULE_CHARS,
        "  fixed: other characters": lambda it: not it["base_right"] and it["char"] not in RULE_CHARS,
        f"kept (right answers of {first})": lambda it: it["base_right"],
        "  kept: 度/自/在": lambda it: it["base_right"] and it["char"] in RULE_CHARS,
        "  kept: other characters": lambda it: it["base_right"] and it["char"] not in RULE_CHARS,
    }
    result = {}
    print(f"\n{'items':34} {'n':>4}  " + "  ".join(f"{n:>14}" for n in answers))
    for label, pick in groups.items():
        sub = [it for it in items if pick(it)]
        row = {}
        for n in answers:
            k = sum(answers[n][it["id"]] == it["truth"] for it in sub)
            row[n] = {"right": k, "n": len(sub), "text": f"{k:3d}/{len(sub):<3d} ({k / max(len(sub), 1):4.0%})"}
        result[label] = row
        print(f"{label:34} {len(sub):4d}  " + "  ".join(f"{row[n]['text']:>14}" for n in answers))
    by_char = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it["char"] in RULE_CHARS:
            for n in answers:
                by_char[(it["char"], it["label"], "agreement check" if it["agree_check"] else "label vs model")][n].append(answers[n][it["id"]] == it["truth"])
    print("\nper rule character (items where the label is the one the rule changes):")
    for key, per in sorted(by_char.items()):
        print(f"  {key[0]} label {key[1]:5} {key[2]:16} " + "  ".join(f"{n}: {sum(v)}/{len(v)}" for n, v in per.items()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"models": args.models, "result": result, "items": [{k: v for k, v in it.items() if k != "gold"} | {"answers": {n: answers[n][it["id"]] for n in answers}} for it in items]},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
