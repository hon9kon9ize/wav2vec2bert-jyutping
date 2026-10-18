"""Dry run of label corrections supported by the listening sessions (nothing is written to any dataset).

    python relabel_rules.py --tojyutping_path <dir>

Step 1 turns the verdicts of `tone_listening.py` into evidence per (character, label -> model) pair: how often the listener
found the model right (or, for the agreement check where label and model say the same, the alternative reading). Step 2 counts
what each rule in RULES would touch: exactly in the full training labels for rules that apply `always`, and in the 20k-clip
decode (x13 for the 260,524 tts clips) for rules that apply only `model` (where the model says the corrected reading).
A rule is a decision of mine from that evidence, not something the script derives: edit RULES after listening to more.
Also writes tmp/label_mining/relabel_dry_run.jsonl, one row per touched syllable of the 20k clips, for spot checks.
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import align  # noqa: E402
from mine_label_disagreements import OUT_DIR, convert  # noqa: E402
from tone_listening import TONE_DIR, wilson  # noqa: E402

# (character, label, corrected reading, when): `always` = every such label, `model` = only where the model says the corrected reading
RULES = [
    ("度", "dou3", "dou6", "always"),
    ("自", "zi3", "zi6", "always"),
    ("在", "zoi3", "zoi6", "always"),
    ("時", "si6", "si4", "model"),
    ("為", "wai6", "wai4", "model"),
    ("到", "dou3", "dou2", "model"),
    ("位", "wai6", "wai2", "model"),
    ("會", "wui5", "wui2", "model"),
    ("會", "wui2", "wui5", "model"),
    # second batch (2026-10-18): the listener judged the model right in 6-10 clips each; only applied where the model says the new reading
    ("市", "si6", "si5", "model"),
    ("市", "si2", "si5", "model"),
    ("意", "ji5", "ji3", "model"),
    ("議", "ji6", "ji5", "model"),
    ("白", "baak3", "baak6", "model"),
    ("無", "mou5", "mou4", "model"),
    ("令", "ling4", "ling6", "model"),
    ("別", "bit3", "bit6", "model"),
    ("大", "daai3", "daai6", "model"),
    ("任", "jam4", "jam6", "model"),
    ("間", "gaan1", "gaan3", "model"),
    ("正", "zing1", "zing3", "model"),
    ("下", "haa6", "haa5", "model"),
    ("企", "kei2", "kei5", "model"),
    ("上", "soeng6", "soeng5", "model"),
    ("嘅", "ge2", "ge3", "model"),
    ("會", "wui5", "wui6", "model"),
    ("會", "wui6", "wui5", "model"),
    ("為", "wai4", "wai6", "model"),
]
TTS_TRAIN_CLIPS = 260_524


def evidence(tone_dir):
    """(char, label, model | '=') -> [judged, label wrong]; '=' is the agreement check (label and model say the same)"""
    data = json.load(open(os.path.join(tone_dir, "samples.json")))
    verdicts = {}
    for line in open(os.path.join(tone_dir, "verdicts.jsonl")):
        r = json.loads(line)
        verdicts[r["id"]] = r["verdict"]
    ev = defaultdict(lambda: [0, 0])
    for g in data["groups"]:
        if g.get("kind") == "audit":  # the audit-sample check is a different population, see audit_listening.py check_summary
            continue
        for s in g["samples"]:
            v = verdicts.get(s["id"])
            if v not in ("A", "B"):
                continue
            other_reading = (v == "A") != s["a_is_label"]  # the listener picked the reading that is not the label's
            key = (s["char"], s["gold"], "=" if g.get("kind") == "agree" else "fresh" if g.get("kind") == "fresh" else s["pred"])
            ev[key][0] += 1
            ev[key][1] += other_reading
    return ev


def rule_evidence(ev, char, frm, to, when):
    judged, wrong = ev.get((char, frm, to), [0, 0])
    for extra in ("=", "fresh") if when == "always" else ("fresh",):  # the agreement check only backs `always` rules
        j2, w2 = ev.get((char, frm, extra), [0, 0])
        judged, wrong = judged + j2, wrong + w2
    return judged, wrong


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tojyutping_path", default=None)
    ap.add_argument("--label_dir", default=OUT_DIR)
    ap.add_argument("--tone_dir", default=TONE_DIR)
    args = ap.parse_args()
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    ev = evidence(args.tone_dir)
    rules = {(c, f, t): w for c, f, t, w in RULES}
    print("1. What the listener found, per character and label -> model pair (judged >= 3):")
    print(f"   {'char':4} {'label':6} {'model':6} {'judged':>6} {'model/alt right':>16} {'95% CI':>9}   rule")
    for (char, frm, to), (judged, wrong) in sorted(ev.items(), key=lambda kv: (-kv[1][0], kv[0])):
        if judged < 3:
            continue
        lo, hi = wilson(wrong, judged)
        when = rules.get((char, frm, to))
        if when is None and to in ("=", "fresh"):  # the agreement check only backs `always` rules, a fresh check backs any rule
            when = next((w for c, f, _, w in RULES if (c, f) == (char, frm) and (to == "fresh" or w == "always")), None)
        mark = f"yes ({when})" if when else "no"
        print(f"   {char:4} {frm:6} {to:6} {judged:6d} {wrong:8d} ({wrong / judged:4.0%}) {lo:4.0%}-{hi:3.0%}   {mark}")
    print("   (model '=' is the agreement check: label and model both say the label's reading; 'fresh' is the rule check on clips outside the 20k; 'right' = the alternative was heard)")

    by_char, _ = pickle.load(open(os.path.join(args.label_dir, "lexicon.pkl"), "rb"))
    clips = [json.loads(line) for line in open(os.path.join(args.label_dir, "decoded.jsonl"))]
    counts, rows = Counter(), []
    wanted = {c for c, _, _, _ in RULES}
    for c in clips:
        conv, chars, ok = convert(tj, c["text"])
        if not ok or len(conv) != len(c["gold"]):
            continue
        ops = {i: (op, g, p) for op, i, g, p in align(c["gold"], c["pred"]) if i is not None}
        for i, (ch, _) in enumerate(chars):
            if ch not in wanted or i not in ops:
                continue
            op, g, p = ops[i]
            model = p if op == "S" else g if op == "=" else None
            counts[(ch, g, model)] += 1
            for char, frm, to, when in RULES:
                if (ch, g) == (char, frm) and (when == "always" or model == to):
                    rows.append({"train_idx": c["train_idx"], "pos": i, "char": ch, "label": g, "to": to, "model": model, "text": c["text"]})
    scale = TTS_TRAIN_CLIPS / len(clips)
    print(f"\n2. What each rule touches ({len(clips)} decoded tts clips; x{scale:.1f} for all tts training clips, emo not sampled):")
    print(f"   {'rule':16} {'when':7} {'evidence':>10}  {'all train labels':>17}  {'in 20k':>7} {'fixed':>6} {'new err':>8}  (for the current model, on the corrected labels)")
    total_touched = total_fixed = total_new = 0
    for char, frm, to, when in RULES:
        judged, wrong = rule_evidence(ev, char, frm, to, when)
        touched = sum(n for (ch, g, _), n in counts.items() if (ch, g) == (char, frm)) if when == "always" else counts[(char, frm, to)]
        fixed, new = counts[(char, frm, to)], counts[(char, frm, frm)] if when == "always" else 0
        full = f"{by_char.get(char, {}).get(frm, 0):,}" if when == "always" else f"~{touched * scale:,.0f} (tts)"
        print(f"   {char + ' ' + frm + '->' + to:16} {when:7} {wrong:>4}/{judged:<5}  {full:>17}  {touched:7d} {fixed:6d} {new:8d}")
        total_touched += touched; total_fixed += fixed; total_new += new
    print(f"   {'total':16} {'':7} {'':>10}  {'':>17}  {total_touched:7d} {total_fixed:6d} {total_new:8d}")
    total_syl = sum(len(c["gold"]) for c in clips)
    print(f"\n   Before retraining the current model scores {total_fixed - total_new:+d} errors on the corrected labels of these {total_syl:,} syllables "
          "(fixed minus new): `new err` are syllables where the model repeats the wrong label, so measured SER rises until a model is retrained on the corrected labels.")
    path = os.path.join(args.label_dir, "relabel_dry_run.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   {len(rows)} touched syllables listed in {path}")


if __name__ == "__main__":
    main()
