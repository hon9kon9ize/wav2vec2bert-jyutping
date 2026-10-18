"""Write new text for rare syllables with a Cantonese masked LM (ModernCantoneseBert-Base).

    python make_rare_sentences.py --residual tmp/synth_plan_v2/residual.json --out tmp/synth_plan_v2/rare_sentences.json

For every rare syllable that the existing sentences cannot cover (select_synth_sentences.py -> residual.json), real sentences
(5-14 syllables, one character per syllable) are taken as frames. A position is masked, the masked LM scores every character that
can be read with the target syllable (syllable inventory from ToJyutping), and the best-fitting character replaces the original;
the label is the frame's jyutping with that syllable replaced by the target one. Two different rare syllables are inserted per
sentence (at least 3 positions apart), so the sentences stay mostly natural and the label of every syllable is exact: the TTS
speaks the phone string it is given, whatever the text means.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

CJK = re.compile(r"[一-鿿]")


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--inventory", default=os.path.join(REPO, "tmp", "synth_plan", "syllable_inventory.json"))
    ap.add_argument("--hkcancor", default=os.path.join(REPO, "tmp", "synth_plan", "hkcancor_utts.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_syllable", type=int, default=8, help="new sentences per rare syllable (each sentence carries two)")
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text as norm

    rng = random.Random(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("hon9kon9ize/ModernCantoneseBert-Base")
    bert = AutoModelForMaskedLM.from_pretrained("hon9kon9ize/ModernCantoneseBert-Base").to(dev).eval()
    vocab = tok.get_vocab()
    inv = json.load(open(args.inventory))["inv"]
    residual = json.load(open(args.residual))

    # frames: real sentences with one character per syllable
    frames = []
    tr = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select(range(6000, 26000)).select_columns(["text", "phone"])
    emo = load_any_dataset(args.emo_path)
    emo = (emo["train"] if hasattr(emo, "keys") else emo).select(range(6000)).select_columns(["text", "labels"])
    srcs = [(t, p) for t, p in zip(tr["text"], tr["phone"])] + [(t, p) for t, p in zip(emo["text"], emo["labels"])]
    srcs += [(u["text"], u["phone"]) for u in json.load(open(args.hkcancor))]
    for text, phone in srcs:
        chars = [c for c in text if CJK.search(c)]
        syl = norm(phone).split()
        if 5 <= len(chars) <= 14 and len(chars) == len(syl) and all(c in vocab for c in chars):
            frames.append((chars, syl))
    rng.shuffle(frames)
    print(f"{len(frames)} frames; {len(residual)} rare syllables to place", flush=True)

    # candidate characters per syllable: those with a token in the LM vocabulary
    cands = {}
    for r in residual:
        cs = [c for c in inv.get(r["syllable"], []) if c in vocab]
        if cs:
            cands[r["syllable"]] = cs
    todo = Counter()
    for r in residual:
        if r["syllable"] in cands:
            todo[r["syllable"]] = min(args.per_syllable, max(r["target"] - r["covered"], 1))
    print(f"{len(cands)} syllables have a usable character; {sum(todo.values())} insertions to make", flush=True)

    @torch.no_grad()
    def best_char(chars, pos, syl):
        """most plausible character at `pos` among those read with `syl`, and its log-probability"""
        ids = tok(" ".join(chars) if False else "".join(chars), return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[1] != len(chars) + 2:
            return None
        ids[0, pos + 1] = tok.mask_token_id
        lp = torch.log_softmax(bert(input_ids=ids).logits[0, pos + 1].float(), -1)
        cs = cands[syl]
        sc = lp[[vocab[c] for c in cs]]
        top = torch.topk(sc, k=min(3, len(cs)))
        k = int(top.indices[rng.randrange(len(top.indices))])
        return cs[k], float(sc[k])

    out, pending = [], list(todo.elements())
    rng.shuffle(pending)
    n_frame = 0
    while len(pending) >= 2 and n_frame < len(frames):
        a, b = pending.pop(), pending.pop()
        if a == b:
            pending.insert(0, b)
            continue
        chars, syl = frames[n_frame]
        n_frame += 1
        chars, syl = list(chars), list(syl)
        L = len(chars)
        pos = []
        for s in (a, b):
            best = None
            for p in rng.sample(range(L), min(L, 5)):
                if any(abs(p - q) < 3 for q in pos):
                    continue
                r = best_char(chars, p, s)
                if r and (best is None or r[1] > best[2]):
                    best = (p, r[0], r[1])
            if best is None:
                break
            p, c, _ = best
            chars[p] = c; syl[p] = s; pos.append(p)
        if len(pos) == 2:
            out.append({"id": f"rare{len(out):05d}", "src": "mlm", "text": "".join(chars), "phone": " ".join(syl), "targets": [a, b]})
        if len(out) % 200 == 0 and out:
            print(f"{len(out)} sentences", flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=0)
    got = Counter(t for r in out for t in r["targets"])
    print(f"wrote {len(out)} sentences covering {len(got)} of {len(todo)} syllables; examples:")
    for r in out[:8]:
        print(f"   {r['text']}   {r['phone']}   targets {r['targets']}")


if __name__ == "__main__":
    main()
