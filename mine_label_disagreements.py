"""Where do the training labels and the model disagree - and who is probably right? (label-correction feasibility)

Step 1 - decode run21 over a random sample of training clips (batch size 1, length-aware decode, resumable):

    python mine_label_disagreements.py decode --n 20000

Step 2 - classify every substituted syllable by what ToJyutping(text) says (needs ToJyutping, see check_labels_vs_tojyutping.py),
print the counts and write a listening page (tmp/label_mining/listen.html, wavs next to it):

    python mine_label_disagreements.py report --tojyutping_path <dir>

Categories of a substitution (label g, model p, converter c, `merger` = differs only by a merger pair):
  agree_conv  p == c != g   model and converter agree against the label (the only relabel candidates)
  model_only  g == c != p   label and converter agree, the model deviates alone (probably a model error)
  all_differ  three different readings
  no_conv     converter unavailable (syllable count differs or a character is unreadable)

The audio has to be judged by ear: the page stores the verdicts in the browser and exports them as JSON.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from check_labels_vs_tojyutping import SYL, part_diff  # noqa: E402
from error_consistency import align, make_predictor  # noqa: E402

OUT_DIR = os.path.join(REPO, "tmp", "label_mining")
CATEGORIES = ("agree_conv", "model_only", "all_differ", "no_conv")


def train_split():
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split

    return make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"]


def cmd_decode(args):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import librosa
    import numpy as np
    import torch
    from datasets import Audio

    from eval_w2v2bert_nucleus_tone import read_audio
    from jyutping import normalize_jyutping_text

    torch.set_num_threads(args.threads)
    train = train_split()
    idx = sorted(random.Random(args.seed).sample(range(len(train)), args.n))
    out_path = os.path.join(args.out_dir, "decoded.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["train_idx"] for line in open(out_path)}
    todo = [i for i in idx if i not in done]
    print(f"{len(idx)} sampled, {len(done)} already decoded, {len(todo)} to go", flush=True)
    ds = train.select(todo).select_columns(["audio", "phone", "text"]).cast_column("audio", Audio(decode=False))
    predict = make_predictor("ours", args.model_id)
    with open(out_path, "a") as f:
        for k, row in enumerate(ds):
            y, sr = read_audio(row["audio"])
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            y = y.astype(np.float32)
            rec = {"train_idx": todo[k], "dur": len(y) / 16000, "text": row["text"],
                   "gold": normalize_jyutping_text(row["phone"]).split(), "pred": predict(y)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 500 == 0:
                f.flush()
                print(f"{k + 1}/{len(todo)}", flush=True)
    print("saved", out_path, flush=True)


def convert(tj, text):
    """text -> (converter syllables, (character, offset in text) each came from, None if a CJK character is unreadable)"""
    syls, chars, offset = [], [], 0
    for ch, jp in tj.get_jyutping_list(text):
        found = SYL.findall(jp) if jp else []
        syls.extend(found)
        chars.extend([(ch, offset)] * len(found))
        offset += len(ch)
        if not found and "一" <= ch <= "鿿":
            return [], [], None
    return syls, chars, True


def classify(clips, tj):
    """-> (substitutions with their category, syllable count, clips with >= 1 agree_conv)"""
    subs, total, clips_agree = [], 0, set()
    for c in clips:
        gold = c["gold"]
        total += len(gold)
        conv, chars, ok = convert(tj, c["text"])
        usable = bool(ok) and len(conv) == len(gold)
        for op, i, g, p in align(gold, c["pred"]):
            if op != "S":
                continue
            cv = conv[i] if usable else None
            cat = "no_conv" if cv is None else "agree_conv" if p == cv else "model_only" if g == cv else "all_differ"
            subs.append({"train_idx": c["train_idx"], "pos": i, "text": c["text"], "char": chars[i][0] if usable else "", "offset": chars[i][1] if usable else -1,
                         "gold": g, "pred": p, "conv": cv, "cat": cat, "merger": part_diff(g, p)[1],
                         "tone_only": part_diff(g, p)[0] == ["tone"]})
            if cat == "agree_conv":
                clips_agree.add(c["train_idx"])
    return subs, total, len(clips_agree)


def write_listening_page(sample, out_dir):
    import soundfile as sf

    from eval_w2v2bert_nucleus_tone import read_audio

    wav_dir = os.path.join(out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    train = train_split()
    rows = []
    for n, s in enumerate(sample):
        name = f"{n:03d}_{s['cat']}_{s['train_idx']}.wav"
        y, sr = read_audio(train.select([s["train_idx"]])[0]["audio"])
        sf.write(os.path.join(wav_dir, name), y, sr)
        s["file"] = name
        text = s["text"]
        marked, k = text, s["offset"]
        if s["char"] and text[k : k + len(s["char"])] == s["char"]:
            marked = f"{text[:k]}<mark>{s['char']}</mark>{text[k + len(s['char']):]}"
        rows.append(
            f"<tr data-id='{s['file']}'><td>{n}</td><td>{s['cat']}{' / merger' if s['merger'] else ''}</td>"
            f"<td><audio controls preload='none' src='wav/{name}'></audio></td><td class='t'>{marked}</td>"
            f"<td>label <b>{s['gold']}</b></td><td>model <b>{s['pred']}</b></td><td>converter <b>{s['conv'] or '-'}</b></td>"
            "<td><label><input type=radio name='r{0}' value=label> label</label> "
            "<label><input type=radio name='r{0}' value=model> model</label> "
            "<label><input type=radio name='r{0}' value=neither> neither/unsure</label></td></tr>".format(n)
        )
    html = f"""<!doctype html><meta charset=utf-8><title>label vs model</title>
<style>body{{font:15px sans-serif;margin:16px}}td{{padding:4px 8px;border-bottom:1px solid #ccc}}.t{{font-size:18px}}mark{{background:#ffd54f}}</style>
<p>Listen, then pick who matches what was said (the highlighted character is the syllable in question). Verdicts are kept in this browser.
<button onclick="dl()">export JSON</button></p><table>{''.join(rows)}</table>
<script>
const K='label_mining_verdicts';let v=JSON.parse(localStorage.getItem(K)||'{{}}');
document.querySelectorAll('tr[data-id]').forEach(tr=>{{const id=tr.dataset.id;
  tr.querySelectorAll('input').forEach(i=>{{if(v[id]===i.value)i.checked=true;
    i.onchange=()=>{{v[id]=i.value;localStorage.setItem(K,JSON.stringify(v))}}}})}});
function dl(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(v,null,1)]));a.download='verdicts.json';a.click()}}
</script>"""
    open(os.path.join(out_dir, "listen.html"), "w").write(html)


def cmd_report(args):
    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    clips = [json.loads(line) for line in open(os.path.join(args.out_dir, "decoded.jsonl"))]
    subs, total, clips_agree = classify(clips, tj)
    n = len(subs)
    print(f"{len(clips)} clips, {total} syllables, {n} substitutions ({n / total:.2%} of syllables)")
    print(f"{'category':12} {'subs':>6} {'% of subs':>10} {'% of syll':>10} {'merger-only':>12} {'tone-only':>10}")
    for cat in CATEGORIES:
        sc = [s for s in subs if s["cat"] == cat]
        print(f"{cat:12} {len(sc):6d} {len(sc) / n:10.1%} {len(sc) / total:10.2%} "
              f"{sum(s['merger'] for s in sc):12d} {sum(s['tone_only'] for s in sc):10d}")
    print(f"clips with >= 1 agree_conv syllable: {clips_agree} ({clips_agree / len(clips):.2%}); "
          f"a relabel of those would touch {sum(s['cat'] == 'agree_conv' for s in subs) / total:.2%} of syllables")
    for cat in ("agree_conv", "model_only"):
        print(f"\nmost frequent {cat} readings (character: label -> model, count):")
        top = Counter((s["char"], s["gold"], s["pred"]) for s in subs if s["cat"] == cat).most_common(25)
        print("  " + "   ".join(f"{c}:{g}->{p} x{k}" for (c, g, p), k in top))
    with open(os.path.join(args.out_dir, "substitutions.jsonl"), "w") as f:
        for s in subs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    rng = random.Random(0)
    sample = []
    for cat in ("agree_conv", "model_only", "all_differ"):
        for merger in (True, False):
            pool = [s for s in subs if s["cat"] == cat and s["merger"] == merger]
            sample += rng.sample(pool, min(args.per_bucket, len(pool)))
    rng.shuffle(sample)
    write_listening_page(sample, args.out_dir)
    print(f"\nlistening page with {len(sample)} syllables: {os.path.join(args.out_dir, 'listen.html')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    d.add_argument("--n", type=int, default=20000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--out_dir", default=OUT_DIR)
    d.set_defaults(fn=cmd_decode)
    r = sub.add_parser("report")
    r.add_argument("--tojyutping_path", default=None)
    r.add_argument("--per_bucket", type=int, default=25, help="syllables sampled per (category, merger) bucket")
    r.add_argument("--out_dir", default=OUT_DIR)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)
