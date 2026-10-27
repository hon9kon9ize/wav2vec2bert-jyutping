"""Write a COMPLETE corrected label file (every clip, not only the changed ones) from a decode and a corrected_labels.json.

    python export_corrected_labels.py --decoded <decoded.jsonl> --corrected <corrected_labels.json> --out <labels.jsonl> [--corpus tts]

One JSON line per clip: corpus, split (train), idx (row of the train split of that corpus), text, phone_original, phone_corrected,
changed (the syllable positions that differ). The datasets themselves are never edited: feed `corrected_labels.json` to
`compare_heads.py --phone_override`, or join this file on (corpus, idx) in your own training script.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoded", required=True)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="tts")
    args = ap.parse_args()
    corrected = {int(k): v for k, v in json.load(open(args.corrected)).items()}
    n = changed_clips = changed_syllables = 0
    rows = []
    for line in open(args.decoded):
        r = json.loads(line)
        new = corrected.get(r["train_idx"])
        new_syllables = new.split() if new else r["gold"]
        assert len(new_syllables) == len(r["gold"]), r["train_idx"]
        diff = [i for i, (a, b) in enumerate(zip(r["gold"], new_syllables)) if a != b]
        rows.append({"corpus": args.corpus, "split": "train", "idx": r["train_idx"], "text": r["text"], "phone_original": " ".join(r["gold"]),
                     "phone_corrected": " ".join(new_syllables), "changed": diff})
        n += 1
        changed_clips += bool(diff)
        changed_syllables += len(diff)
    rows.sort(key=lambda r: r["idx"])
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.corpus}: {n} clips written, {changed_clips} clips / {changed_syllables} syllables corrected -> {args.out}")


if __name__ == "__main__":
    main()
