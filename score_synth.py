"""Score a model on synthesized clips (synth_tts.py output) against the jyutping they were synthesized from, and compare
with the same model's errors on the real clips of the same sentences when a predictions file is given.

    python score_synth.py --synth_dir tmp/synth/test40 [--model_id checkpoints/run21_40000] [--preds tmp/preds_run21.json]
"""

import argparse
import json
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import TONE_MERGERS, align, make_predictor, parse  # noqa: E402


def errs(gold, pred):
    e = m = 0
    for op, _, r, h in align(gold, pred):
        if op != "=":
            e += 1
            if op == "S":
                po, pn, pc, pt = parse(r)
                ho, hn, hc, ht = parse(h)
                m += (po, pn, pc) == (ho, hn, hc) and pt != ht and frozenset((pt, ht)) in TONE_MERGERS
    return e, m


def main():
    os.environ.setdefault("USE_TF", "0")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth_dir", required=True)
    ap.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    ap.add_argument("--preds", default=os.path.join(REPO, "tmp", "preds_run21.json"))
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    import librosa
    import numpy as np

    from jyutping import normalize_jyutping_text

    man = json.load(open(os.path.join(args.synth_dir, "manifest.json")))
    predict = make_predictor("ours", args.model_id)
    real = {}
    if os.path.exists(args.preds):
        for c in json.load(open(args.preds)):
            real[("val" if c["set"] == "tts_val" else "test" if c["set"] == "tts_test" else c["set"]) + f"{c['idx']:03d}"] = c
    tot = Counter()
    by_ref = {}
    shown = 0
    for it in man:
        y = librosa.load(os.path.join(args.synth_dir, it["file"]), sr=16000, mono=True)[0].astype(np.float32)
        gold = normalize_jyutping_text(it["phone"]).split()
        pred = predict(y)
        e, m = errs(gold, pred)
        r = by_ref.setdefault(it["ref"], Counter())
        r["syl"] += len(gold); r["err"] += e; r["merger"] += m; r["clips"] += 1; r["exact"] += e == 0
        tot["syl"] += len(gold); tot["err"] += e; tot["merger"] += m; tot["dur"] += it["dur"]
        rc = real.get(it["id"])
        if rc:
            re_, rm = errs(rc["gold"], rc["pred"])
            tot["real_syl"] += len(rc["gold"]); tot["real_err"] += re_; tot["real_merger"] += rm; tot["paired"] += 1
        if e and shown < args.show:
            shown += 1
            diffs = [f"{a}->{b}" for op, _, a, b in align(gold, pred) if op == "S"] + \
                    [f"{a}->(del)" for op, _, a, b in align(gold, pred) if op == "D"] + \
                    [f"(ins)->{b}" for op, _, a, b in align(gold, pred) if op == "I"]
            print(f"  {it['file']:28} {e} err: {diffs}" + (f"   | real clip: {errs(rc['gold'], rc['pred'])[0]} err" if rc else ""))
    print(f"\nsynthetic: {len(man)} clips, {tot['dur'] / len(man):.2f} s mean, {tot['syl']} syllables, SER {100 * tot['err'] / tot['syl']:.2f}% "
          f"({tot['err']} errors, {tot['merger']} merger-pair tone)")
    for ref, r in by_ref.items():
        print(f"   voice {ref:12} clips {r['clips']:4}  SER {100 * r['err'] / r['syl']:.2f}%  exact clips {100 * r['exact'] / r['clips']:.0f}%")
    if tot["paired"]:
        print(f"real clips of the same {tot['paired']} sentences: SER {100 * tot['real_err'] / tot['real_syl']:.2f}% ({tot['real_err']} errors, {tot['real_merger']} merger-pair tone)")


if __name__ == "__main__":
    main()
