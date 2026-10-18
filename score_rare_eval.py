"""Score single-head checkpoints on the synthetic rare-syllable eval set (JyutVoice audio, ToJyutping labels).

    python score_rare_eval.py checkpoints/run21_40000 [checkpoints/run20_best ...] [--tojyutping_path <dir>]
        [--eval_dir /home/joseph/projects/qwen3asr-analysis/synthetic/rare_eval] [--out tmp/rare_eval_preds.json] [--show 15]

The set has the Chinese `text` per clip (`eval.jsonl`) and, in `manifest.jsonl`, `targets` (the rare words each sentence was built around)
and `g2p_fix` ("char:ToJyutping reading->reading the TTS was given"). Two labels per clip are scored:
  tojyutping = ToJyutping(text)                              what you get by converting the text
  spoken     = the same with every g2p_fix applied            what the TTS was asked to say (the primary number)
The audio is synthetic, and the TTS inherits reading noise (it can say another reading than the one it was given), so both are
exact only for the phone string, not for the audio.
"""

import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from check_labels_vs_tojyutping import part_diff  # noqa: E402
from error_consistency import align, make_predictor  # noqa: E402

CJK = re.compile(r"[一-鿿]")
SYL = re.compile(r"[a-z]+[1-6]")
FIX = re.compile(r"^(.+):([a-z]+[1-6])->([a-z]+[1-6])$")
DEFAULT_DIR = "/home/joseph/projects/qwen3asr-analysis/synthetic/rare_eval"


def read_text(tj, text):
    """text -> ([(char position, syllable)] for every character ToJyutping reads, number of CJK characters it could not read)"""
    items, missing, pos = [], 0, 0
    for ch, jp in tj.get_jyutping_list(text):
        syls = SYL.findall(jp) if jp else []
        items.extend((pos, s) for s in syls)
        if not syls and CJK.search(ch):
            missing += 1
        pos += len(ch)
    return items, missing


def apply_fixes(text, items, fixes):
    """Apply "char:old->new" fixes: every occurrence of the character whose reading is `old` gets `new`. Returns (items, applied, unmatched)."""
    items, applied, unmatched = list(items), 0, []
    for fix in fixes:
        m = FIX.match(fix)
        char, old, new = m.groups()
        hit = [i for i, (pos, syl) in enumerate(items) if text[pos:pos + len(char)] == char and syl == old]
        for i in hit:
            items[i] = (items[i][0], new)
        applied += len(hit)
        if not hit:
            unmatched.append(fix)
    return items, applied, unmatched


def target_indices(text, items, targets):
    """syllable indices (into `items`) of the characters of the target words"""
    index_of = {pos: i for i, (pos, _) in enumerate(items)}
    out = set()
    for t in targets:
        start = text.find(t)
        while start >= 0:
            out.update(index_of[p] for p in range(start, start + len(t)) if p in index_of)
            start = text.find(t, start + 1)
    return out


def score(ref, pred, targets):
    ops = align(ref, pred)
    err = [op for op, *_ in ops if op != "="]
    tgt_err = sum(1 for op, i, *_ in ops if op != "=" and i in targets)
    return ops, len(err), tgt_err


def main():
    os.environ.setdefault("USE_TF", "0")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--eval_dir", default=DEFAULT_DIR)
    ap.add_argument("--tojyutping_path", default=None, help="a directory that contains the ToJyutping package")
    ap.add_argument("--out", default=os.path.join(REPO, "tmp", "rare_eval_preds.json"))
    ap.add_argument("--show", type=int, default=15, help="most frequent confusions to list")
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import librosa
    import numpy as np
    import ToJyutping as tj

    manifest = [json.loads(line) for line in open(os.path.join(args.eval_dir, "manifest.jsonl"), encoding="utf-8")]
    clips, missing_total, unmatched_total = [], 0, []
    for m in manifest:
        items, missing = read_text(tj, m["text"])
        fixed, applied, unmatched = apply_fixes(m["text"], items, m.get("g2p_fix", []))
        missing_total += missing
        unmatched_total += [f"{m['id']} {u}" for u in unmatched]
        clips.append({
            "id": m["id"], "text": m["text"], "spk": m["spk"], "fixed": bool(m.get("g2p_fix")),
            "plain": [s for _, s in items], "spoken": [s for _, s in fixed],
            "targets": sorted(target_indices(m["text"], items, m["targets"])),
            "audio": librosa.load(os.path.join(args.eval_dir, m["wav"]), sr=16000, mono=True)[0].astype(np.float32),
        })
    n_syl = sum(len(c["spoken"]) for c in clips)
    n_tgt = sum(len(c["targets"]) for c in clips)
    print(f"{len(clips)} clips, {n_syl} syllables, {n_tgt} in target words, {sum(c['fixed'] for c in clips)} clips with a g2p_fix; "
          f"{missing_total} CJK characters ToJyutping could not read; {len(unmatched_total)} fixes matched no character {unmatched_total[:5] or ''}")

    saved = {c["id"]: {"text": c["text"], "plain": c["plain"], "spoken": c["spoken"], "pred": {}} for c in clips}
    for model in args.models:
        predict = make_predictor("ours", model)
        tot = collections.Counter()
        confusions, merger_only, subs = collections.Counter(), 0, 0
        for c in clips:
            pred = predict(c["audio"])
            saved[c["id"]]["pred"][model] = pred
            tgt = set(c["targets"])
            ops, e, te = score(c["spoken"], pred, tgt)
            _, e_plain, te_plain = score(c["plain"], pred, tgt)
            grp = "fixed" if c["fixed"] else "unfixed"
            tot["err"] += e; tot["tgt_err"] += te; tot["err_plain"] += e_plain; tot["tgt_err_plain"] += te_plain
            tot["exact"] += e == 0; tot[f"err_{grp}"] += e; tot[f"n_{grp}"] += len(c["spoken"]); tot[f"clips_{grp}"] += 1
            for op, i, a, b in ops:
                if op == "S":
                    subs += 1
                    confusions[(a, b)] += 1
                    merger_only += part_diff(a, b)[1]
        pct = lambda k, n: f"{tot[k]}/{n} = {100 * tot[k] / n:.2f}%"
        print(f"\n{model}")
        print(f"  SER spoken label      {pct('err', n_syl)}   (clips exactly right {tot['exact']}/{len(clips)})")
        print(f"  SER plain ToJyutping  {pct('err_plain', n_syl)}")
        print(f"  target-word syllables {pct('tgt_err', n_tgt)}   other syllables {tot['err'] - tot['tgt_err']}/{n_syl - n_tgt} = "
              f"{100 * (tot['err'] - tot['tgt_err']) / (n_syl - n_tgt):.2f}%")
        print(f"  clips without fix {pct('err_unfixed', tot['n_unfixed'])}   with fix {pct('err_fixed', tot['n_fixed'])}")
        print(f"  substitutions that differ only by a merger pair (tone 3/6 4/6 2/5, onset n/l ng/- gw/g): {merger_only}/{subs}")
        print(f"  most frequent confusions (label->model): " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in confusions.most_common(args.show)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(saved, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nper-clip labels and predictions: {args.out}")


if __name__ == "__main__":
    main()
