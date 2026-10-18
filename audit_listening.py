"""Random audit of the training labels at clip level: what share of the label syllables is wrong? (unbiased: random clips)

Every earlier listening check looked at clips where the model and the label disagree, or at chosen characters; the 99% of syllables
where they agree were never sampled. This audit draws clips uniformly at random from the whole tts training split (the model's own
prediction is not shown, to avoid anchoring), so the share of wrong syllables is an honest estimate of the label noise.

    python audit_listening.py build   --tojyutping_path <dir> [--n 150] [--seed 0]   # wavs + CTC syllable windows -> tmp/label_mining/audit/
    python audit_listening.py pitch                                                  # F0 tracks (needs praat-parselmouth)
    python audit_listening.py serve   [--port 8778]                                  # http://127.0.0.1:8778, verdicts -> audit/verdicts.jsonl
    python audit_listening.py summary                                                # error rate, visible vs invisible errors, coverage
    python audit_listening.py check --tojyutping_path <dir>                          # blind A/B page for the audit sample's rule-character syllables
    python tone_listening.py serve --out_dir tmp/label_mining/audit_check --port 8780   # ... then listen; afterwards:
    python audit_listening.py check_summary

On the page: click a syllable chip (or the picture) to hear it, mark it wrong and type what you hear (a toned jyutping such as dou6,
`?` if you cannot tell, `-` if the label has a syllable that is not spoken); put anything else (a spoken syllable missing from the
label, bad audio) in the note. "All correct" confirms a clip, "Save errors" saves the marked ones, "Unusable" skips a clip.
Syllable windows come from the CTC spikes of run21 (approximate); the whole-clip button always plays everything.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import align  # noqa: E402
from mine_label_disagreements import OUT_DIR, convert  # noqa: E402
from tone_listening import Handler, frame_syllables, wilson  # noqa: E402

AUDIT_DIR = os.path.join(OUT_DIR, "audit")
STATUSES = ("ok", "errors", "unusable")


# ----------------------------------------------------------------------------- build
def syllable_windows(spikes, dur, gold, pred):
    """per gold syllable: (window, predicted syllable) or (None, None); windows as in ../wav2vec2-jyutping-playground: a syllable ends at
    the end of the frame of its rime spike (frame k is centred at 0.03 + 0.04 k s) and starts where the previous one ended"""
    ends = [min(dur, 0.05 + 0.04 * last) for _, _, last in spikes]
    matched, j = {}, 0
    for op, i, _, _ in align(gold, pred):
        if op in ("=", "S"):
            matched[i] = j
        if op in ("=", "S", "I"):
            j += 1
    out = []
    for i in range(len(gold)):
        k = matched.get(i)
        if k is None:
            out.append((None, None))
            continue
        end = ends[k]
        start = max(0.0, end - 0.25) if k == 0 else ends[k - 1]
        start = max(start, end - 0.6)
        end = min(max(end, start + 0.05), dur)
        out.append(([round(start, 3), round(end, 3)], pred[k]))
    return out


def cmd_build(args):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor

    from ctc_utils import load_ctc_tokenizer
    from eval_w2v2bert_nucleus_tone import read_audio
    from mine_label_disagreements import train_split
    from model import Wav2Vec2BertForJyutpingCTC

    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    rows = [json.loads(line) for line in open(args.decoded_train)]
    picked = sorted(random.Random(args.seed).sample(rows, args.n), key=lambda r: r["train_idx"])
    print(f"{args.n} clips drawn uniformly from {len(rows)} training clips (seed {args.seed})", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_ctc_tokenizer(os.path.join(args.model_id, "vocab.json"))
    proc = Wav2Vec2BertProcessor(feature_extractor=SeamlessM4TFeatureExtractor.from_pretrained(args.model_id), tokenizer=tok)
    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        args.model_id, add_adapter=True, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=tok.pad_token_id, vocab_size=len(tok),
    ).to(dev).eval()
    train = train_split()
    wav_dir = os.path.join(args.out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    clips = []
    for k, r in enumerate(picked):
        row = train.select([r["train_idx"]]).cast_column("audio", Audio(decode=False))[0]
        y, sr = read_audio(row["audio"])
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        y = y.astype(np.float32)
        sf.write(os.path.join(wav_dir, f"{r['train_idx']}.wav"), y, 16000, subtype="PCM_16")
        inp = proc(y, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(input_features=inp.input_features.to(dev), attention_mask=inp.attention_mask.to(dev), return_dict=True).logits
        n = int(model._get_feat_extract_output_lengths(inp.attention_mask.sum(-1))[0])
        spikes = frame_syllables(tok, logits[0].argmax(-1).tolist()[:n])
        pred = [s for s, _, _ in spikes]
        dur = len(y) / 16000
        conv, chars, ok = convert(tj, r["text"])
        usable = bool(ok) and len(conv) == len(r["gold"])
        windows = syllable_windows(spikes, dur, r["gold"], pred)
        n_bins = 240
        peaks = np.abs(y[: len(y) // n_bins * n_bins]).reshape(n_bins, -1).max(1)
        clips.append({
            "id": r["train_idx"], "text": r["text"], "dur": dur,
            "peaks": [int(round(100 * p / max(float(peaks.max()), 1e-6))) for p in peaks],
            "syllables": [{"i": i, "label": g, "char": chars[i][0] if usable else "", "model": m, "win": w}
                          for i, (g, (w, m)) in enumerate(zip(r["gold"], windows))]})
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{args.n} clips", flush=True)
    json.dump({"model": args.model_id, "seed": args.seed, "population": len(rows), "clips": clips},
              open(os.path.join(args.out_dir, "audit.json"), "w"), ensure_ascii=False)
    n_syl = sum(len(c["syllables"]) for c in clips)
    print(f"wrote {len(clips)} clips, {n_syl} label syllables ({sum(s['win'] is None for c in clips for s in c['syllables'])} without a window) "
          f"-> {args.out_dir}", flush=True)


def cmd_pitch(args):
    """Praat F0 track (the playground's settings) and pitch range of every clip, into audit.json"""
    import wave

    import numpy as np
    import parselmouth

    path = os.path.join(args.out_dir, "audit.json")
    data = json.load(open(path))
    pitch = {}
    for clip in data["clips"]:
        with wave.open(os.path.join(args.out_dir, "wav", f"{clip['id']}.wav")) as w:
            rate, frames = w.getframerate(), w.readframes(w.getnframes())
        y = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768
        track = parselmouth.Sound(y, sampling_frequency=rate).to_pitch(time_step=0.01, pitch_floor=70.0, pitch_ceiling=450.0)
        f0 = track.selected_array["frequency"]
        voiced = f0[f0 > 0]
        rng = None
        if len(voiced):
            low, high = (float(v) for v in np.percentile(voiced, [10, 90]))
            if high / low < 2 ** (4 / 12):
                centre, half = (low * high) ** 0.5, 2 ** (4 / 24)
                low, high = centre / half, centre * half
            rng = [round(low, 1), round(high, 1)]
        pitch[str(clip["id"])] = {"t0": round(float(track.xs()[0]), 4), "step": 0.01, "range": rng,
                                  "f0": [round(float(v), 1) if v > 0 else None for v in f0]}
    data["pitch"] = pitch
    json.dump(data, open(path, "w"), ensure_ascii=False)
    print(f"pitch track for {len(pitch)} clips -> {path}")


# ----------------------------------------------------------------------------- serve
def latest_verdicts(audit_dir):
    out, path = {}, os.path.join(audit_dir, "verdicts.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


class AuditHandler(Handler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/data":
            return self._send(200, open(os.path.join(self.tone_dir, "audit.json"), "rb").read(), "application/json")
        if path == "/api/verdicts":
            return self._send(200, json.dumps(latest_verdicts(self.tone_dir), ensure_ascii=False).encode(), "application/json")
        super().do_GET()  # /wav/<id>.wav and /ref/<syllable>.mp3 (the reference voice, cached in <dir>/ref)

    def do_POST(self):
        if self.path != "/api/verdict":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            assert isinstance(body.get("id"), int) and body.get("status") in STATUSES
            wrong = body.get("wrong") or {}
            assert isinstance(wrong, dict) and all(re.fullmatch(r"\d+", k) and isinstance(v, str) and len(v) <= 20 for k, v in wrong.items())
            note = str(body.get("note", ""))[:500]
        except Exception:
            return self._send(400, b"bad request", "text/plain")
        with self.lock, open(os.path.join(self.tone_dir, "verdicts.jsonl"), "a") as f:
            f.write(json.dumps({"id": body["id"], "status": body["status"], "wrong": wrong, "note": note, "ts": time.time()}, ensure_ascii=False) + "\n")
        self._send(200, b"{}", "application/json")


def cmd_serve(args):
    from http.server import ThreadingHTTPServer

    AuditHandler.tone_dir = args.out_dir
    server = ThreadingHTTPServer((args.host, args.port), AuditHandler)
    print(f"serving {args.out_dir} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


# ----------------------------------------------------------------------------- summary
def cmd_summary(args):
    data = json.load(open(os.path.join(args.out_dir, "audit.json")))
    verdicts = latest_verdicts(args.out_dir)
    changes = {}
    path = os.path.join(args.label_dir, "relabel_full_v2", "changes.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            changes[(r["train_idx"], r["pos"])] = r
    clips = {str(c["id"]): c for c in data["clips"]}
    status = Counter(v["status"] for k, v in verdicts.items() if k in clips)
    print(f"{len(data['clips'])} random clips drawn from {data['population']} tts training clips; judged {sum(status.values())}: "
          f"{status['ok']} all correct, {status['errors']} with errors, {status['unusable']} unusable")
    wrong, n_syl, judged_clips = [], 0, 0
    wrong_cover = Counter()
    false_change = []
    for k, v in verdicts.items():
        c = clips.get(k)
        if c is None or v["status"] == "unusable":
            continue
        judged_clips += 1
        n_syl += len(c["syllables"])
        for s in c["syllables"]:
            heard = v["wrong"].get(str(s["i"]))
            change = changes.get((c["id"], s["i"]))
            if heard is not None:
                wrong.append((c, s, heard, change))
            elif change is not None:
                false_change.append((c, s, change))  # a v2 correction on a syllable the listener found correct
    k = len(wrong)
    lo, hi = wilson(k, n_syl)
    print(f"\nlabel syllables judged: {n_syl}; wrong: {k} = {k / max(n_syl, 1):.2%} (95% CI {lo:.2%} to {hi:.2%}, treating syllables as independent)")
    with_err = sum(1 for k2, v in verdicts.items() if k2 in clips and v["status"] == "errors")
    print(f"clips with at least one wrong label syllable: {with_err} of {judged_clips} = {with_err / max(judged_clips, 1):.0%}")
    if wrong:
        visible = [w for w in wrong if w[1]["model"] != w[1]["label"]]
        print(f"\nof the {k} wrong labels, the model (run21) disagreed with the label in {len(visible)} and repeated the wrong label in "
              f"{k - len(visible)} ({(k - len(visible)) / k:.0%}): those are invisible to a disagreement analysis")
        covered = [w for w in wrong if w[3] is not None]
        right_fix = [w for w in covered if w[3]["new"] == w[2]]
        print(f"already changed by the v2 corrections: {len(covered)} ({len(right_fix)} to exactly the reading the listener typed); "
              f"not covered: {k - len(covered)}")
        print(f"v2 corrections on syllables the listener found correct (the correction would have broken a right label): {len(false_change)}")
        print("\nwrong labels (char: label -> heard; model says; covered by a v2 correction?):")
        for c, s, heard, change in sorted(wrong, key=lambda w: (w[1]["char"], w[1]["label"])):
            print(f"  {c['id']:>7} #{s['i']:<2} {s['char'] or '?'}: {s['label']} -> {heard:6} model {s['model']}  "
                  f"{'corrected to ' + change['new'] if change else 'not covered'}")
    notes = [(k2, v["note"]) for k2, v in verdicts.items() if k2 in clips and v.get("note")]
    if notes:
        print("\nnotes:")
        for k2, n in notes:
            print(f"  clip {k2}: {n}")


# ----------------------------------------------------------------------------- blind A/B check of the audit sample
CHECK_DIR = os.path.join(OUT_DIR, "audit_check")


def cmd_check(args):
    """samples.json (the format of tone_listening.py) for the audit sample's syllables that carry a relabel rule's OLD reading: A/B between the
    label's reading and the rule's reading. Group 1 = the rules would change the label; group 2 = the model agrees with the label."""
    import random

    from relabel_rules import RULES

    if args.tojyutping_path:
        sys.path.insert(0, args.tojyutping_path)
    import ToJyutping as tj

    data = json.load(open(os.path.join(args.audit_dir, "audit.json")))
    changes = {}
    for line in open(os.path.join(args.label_dir, "relabel_full_v2", "changes.jsonl")):
        r = json.loads(line)
        changes[(r["train_idx"], r["pos"])] = r["new"]
    changed, agreeing = [], []
    for clip in data["clips"]:
        conv, chars, ok = convert(tj, clip["text"])
        for s in clip["syllables"]:
            variants = [(to, when) for char, frm, to, when in RULES if (char, frm) == (s["char"], s["label"])]
            for v, (to, when) in enumerate(variants):
                if not s["win"] or not ok or s["i"] >= len(chars):
                    continue
                win, dur = s["win"], clip["dur"]
                sample = {"id": f"{clip['id']}_{s['i'] + 1000 * v}", "clip": clip["id"], "text": clip["text"], "char": s["char"], "offset": chars[s["i"]][1],
                          "gold": s["label"], "pred": to, "conv": conv[s["i"]], "taxonomy": "audit_check", "kind": "audit",
                          "a_is_label": random.Random(f"{clip['id']}-{s['i']}-{to}").random() < 0.5, "dur": dur, "shade": win,
                          "seg": [round(max(0.0, win[0] - 0.3), 3), round(min(dur, win[1] + 0.3), 3)], "peaks": clip["peaks"]}
                if changes.get((clip["id"], s["i"])) == to:
                    sample["note"] = "the v2 rules change this label"
                    changed.append(sample)
                else:
                    sample["note"] = f"the model says {s['model']}" + (" (agrees with the label)" if s["model"] == s["label"] else "")
                    agreeing.append(sample)
    groups = [{"gold": "label", "pred": "alternative", "n_total": len(changed), "kind": "audit", "samples": changed},
              {"gold": "label", "pred": "alternative", "n_total": len(agreeing), "kind": "audit", "samples": agreeing}]
    os.makedirs(args.out_dir, exist_ok=True)
    wav = os.path.join(args.out_dir, "wav")
    if not os.path.exists(wav):
        os.symlink(os.path.abspath(os.path.join(args.audit_dir, "wav")), wav)
    keep = {str(s["clip"]) for g in groups for s in g["samples"]}
    json.dump({"model": data["model"], "seed": data["seed"], "clips": len(keep), "groups": groups,
               "pitch": {k: v for k, v in data.get("pitch", {}).items() if k in keep}},
              open(os.path.join(args.out_dir, "samples.json"), "w"), ensure_ascii=False)
    print(f"{len(changed)} syllables the rules would change, {len(agreeing)} with the rule's old reading where the model agrees with the label, "
          f"from {len(keep)} audit clips -> {args.out_dir}")


def cmd_check_summary(args):
    data = json.load(open(os.path.join(args.out_dir, "samples.json")))
    verdicts = {}
    path = os.path.join(args.out_dir, "verdicts.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            verdicts[r["id"]] = r["verdict"]
    for name, g in zip(("the v2 rules would change the label", "old reading, model agrees with the label"), data["groups"]):
        got = Counter()
        rows = []
        for s in g["samples"]:
            v = verdicts.get(s["id"])
            if v is None:
                continue
            who = v if v in ("either", "neither") else ("label" if (v == "A") == s["a_is_label"] else "alternative")
            got[who] += 1
            rows.append((s["char"], s["gold"], s["pred"], who, s["id"], s["text"]))
        n = got["label"] + got["alternative"]
        lo, hi = wilson(got["alternative"], n) if n else (float("nan"), float("nan"))
        print(f"\n{name}: judged {sum(got.values())} of {len(g['samples'])}: label right {got['label']}, alternative right {got['alternative']}, "
              f"either {got['either']}, neither {got['neither']}" + (f"  (alternative right {got['alternative'] / n:.0%}, 95% CI {lo:.0%}-{hi:.0%})" if n else ""))
        for ch, old, new, who, i, text in sorted(rows):
            print(f"   {ch} {old} vs {new}: {who:11} clip {i}  {text}")


# ----------------------------------------------------------------------------- page
PAGE = r"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Label audit</title>
<style>
:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--card:#f4f4f6;--line:#d9d9de;--acc:#2563eb;--ok:#15803d;--bad:#dc2626;--hl:#fde68a}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa3;--card:#1f1f23;--line:#33333a;--acc:#6ea0ff;--ok:#4ade80;--bad:#f87171;--hl:#6b5a14}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.mut{color:var(--mut)}.legend{font-size:13px;color:var(--mut)}
main{max-width:1100px;margin:0 auto;padding:16px}
.card{background:var(--card);border-radius:12px;padding:14px}
#chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.chip{border:2px solid var(--line);background:var(--bg);border-radius:10px;padding:4px 10px;text-align:center;cursor:pointer;min-width:58px}
.chip .ch{font-size:24px;line-height:1.1}.chip .jp{font-size:13px;color:var(--mut)}.chip .fix{font-size:13px;font-weight:700;color:var(--bad);min-height:17px}
.chip.sel{border-color:var(--acc)}.chip.bad{border-color:var(--bad)}.chip.bad .jp{color:var(--bad)}
canvas{width:100%;height:260px;display:block;background:var(--bg);border-radius:8px;cursor:pointer}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
button{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.go{background:var(--acc);border-color:var(--acc);color:#fff}button.ok{border-color:var(--ok);color:var(--ok)}button.bad{border-color:var(--bad);color:var(--bad)}
input[type=text]{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:5px 8px}
#fix{width:270px}#note{flex:1;min-width:200px}
.txt{font-size:20px;margin-bottom:2px}.tag{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--line)}
</style>
<header><b>Label audit</b><span id=prog class=mut></span><button id=nextun>next unjudged (n)</button>
<span class=legend>tones: 1 high level 55 · 2 high rising 25 · 3 mid level 33 · 4 low falling 21 · 5 low rising 23 · 6 low level 22</span>
<span class=legend>keys: o clip · p syllable · ←/→ syllable · r reference voice · x mark wrong · 1 all correct · 2 save errors · 3 unusable · b/n prev/next clip</span></header>
<main><div class=card>
<div class=row style="margin:0 0 6px"><span id=pos class=mut></span><span id=state class=tag></span></div>
<div class=txt id=txt></div>
<div id=chips></div>
<canvas id=cv></canvas><audio id=au preload=auto></audio>
<div class=row><button id=bclip>▶ clip (o)</button><button id=bsyl>▶ syllable (p)</button><button id=bref>▶ reference voice (r)</button>
<span class=mut>selected:</span><b id=selname>-</b><button id=bx>✗ mark wrong (x)</button>
<input type=text id=fix placeholder="heard, e.g. dou6 · ? unsure · - not spoken" disabled><span id=reftyped></span></div>
<div class=row><input type=text id=note placeholder="note: a spoken syllable missing from the label, bad audio, ..."></div>
<div class=row><button class=ok id=bok>✓ all correct (1)</button><button class=bad id=berr>save marked errors (2)</button><button id=bun>unusable (3)</button>
<button id=prev>← previous clip (b)</button><button id=next>next clip (n) →</button></div>
</div></main>
<script>
let D, V = {}, cur = 0, sel = 0, marks = {}, raf = 0, stopAt = null;
const $ = s => document.querySelector(s), au = $('#au');
const TONE_COL = {1:'#d55e00', 2:'#e69f00', 3:'#009e73', 4:'#0072b2', 5:'#cc79a7', 6:'#56b4e9'};
const CHAO = {1:[5,5], 2:[2,5], 3:[3,3], 4:[2,1], 5:[2,3], 6:[2,2]}, GUT = 38;
const levelHz = (l, lo, hi) => lo * Math.pow(hi / lo, (l - 1) / 4);
const clip = () => D.clips[cur];
async function init(){ D = await (await fetch('/api/data')).json(); V = await (await fetch('/api/verdicts')).json();
  $('#bclip').onclick = () => playRange(0, clip().dur); $('#bsyl').onclick = playSyl; $('#bref').onclick = () => ref(clip().syllables[sel].label);
  $('#bx').onclick = toggleWrong; $('#bok').onclick = () => save('ok'); $('#berr').onclick = () => save('errors'); $('#bun').onclick = () => save('unusable');
  $('#prev').onclick = () => go(cur - 1); $('#next').onclick = () => go(cur + 1); $('#nextun').onclick = nextUnjudged;
  $('#fix').onkeydown = e => { if (e.key === 'Enter'){ commitFix(); e.preventDefault(); } e.stopPropagation(); };
  $('#note').onkeydown = e => e.stopPropagation(); addEventListener('resize', () => draw(null));
  cv.onclick = e => { const r = cv.getBoundingClientRect(), t = Math.max(0, ((e.clientX - r.left) - GUT) / (r.width - GUT) * clip().dur);
    const i = clip().syllables.findIndex(s => s.win && t >= s.win[0] && t <= s.win[1]); if (i >= 0) select(i); playRange(t, clip().dur); };
  go(firstUnjudged()); }
const cv = $('#cv');
const judged = id => V[String(id)] !== undefined;
function firstUnjudged(){ const i = D.clips.findIndex(c => !judged(c.id)); return i < 0 ? 0 : i; }
function nextUnjudged(){ for (let k = 1; k <= D.clips.length; k++){ const i = (cur + k) % D.clips.length; if (!judged(D.clips[i].id)) return go(i); } }
function go(i){ if (i < 0 || i >= D.clips.length) return; stop(); cur = i; sel = 0; const v = V[String(clip().id)];
  marks = v ? {...v.wrong} : {}; $('#note').value = v ? (v.note || '') : ''; render(); }
function render(){ const c = clip(); $('#pos').textContent = `clip ${cur + 1} / ${D.clips.length} · ${c.dur.toFixed(1)} s · id ${c.id}`;
  $('#txt').textContent = c.text; const box = $('#chips'); box.innerHTML = '';
  c.syllables.forEach(s => { const d = document.createElement('div'); d.className = 'chip'; d.dataset.i = s.i;
    d.innerHTML = '<div class=ch></div><div class=jp></div><div class=fix></div>'; d.children[0].textContent = s.char || '·'; d.children[1].textContent = s.label;
    d.onclick = () => { select(s.i); playSyl(); }; box.appendChild(d); }); paintChips(); select(sel, true); status(); progress(); }
function paintChips(){ clip().syllables.forEach(s => { const d = $(`.chip[data-i="${s.i}"]`), m = marks[String(s.i)];
  d.classList.toggle('bad', m !== undefined); d.classList.toggle('sel', s.i === sel); d.children[2].textContent = m !== undefined ? '→ ' + m : ''; }); }
function select(i, quiet){ sel = i; paintChips(); const s = clip().syllables[i]; $('#selname').textContent = `${s.char || '·'} ${s.label}`;
  const m = marks[String(i)]; $('#fix').disabled = m === undefined; $('#fix').value = m === undefined ? '' : m; draw(null); }
function toggleWrong(){ const k = String(sel); if (marks[k] === undefined){ marks[k] = '?'; $('#fix').disabled = false; $('#fix').value = ''; $('#fix').focus(); }
  else { delete marks[k]; $('#fix').disabled = true; $('#fix').value = ''; } paintChips(); }
function commitFix(){ const k = String(sel); if (marks[k] === undefined) return; const t = $('#fix').value.trim(); marks[k] = t === '' ? '?' : t; paintChips(); $('#fix').blur();
  if (/^[a-z]+[1-6]$/.test(marks[k])) $('#reftyped').innerHTML = `<button id=bref2>▶ ${marks[k]}</button>`, $('#bref2').onclick = () => ref(marks[k]); else $('#reftyped').innerHTML = ''; }
function status(){ const v = V[String(clip().id)], t = $('#state'); t.textContent = v ? ({ok: 'judged: all correct', errors: `judged: ${Object.keys(v.wrong).length} wrong`, unusable: 'judged: unusable'})[v.status] : 'not judged'; }
function progress(){ const n = D.clips.filter(c => judged(c.id)).length, e = Object.values(V).filter(v => v.status === 'errors').length;
  $('#prog').textContent = `judged ${n} / ${D.clips.length} · clips with errors: ${e}`; }
async function save(st){ commitFix(); const body = {id: clip().id, status: st, wrong: st === 'errors' ? marks : {}, note: $('#note').value};
  if (st === 'errors' && !Object.keys(marks).length) return alert('mark at least one syllable wrong, or press "all correct"');
  const r = await fetch('/api/verdict', {method: 'POST', body: JSON.stringify(body)}); if (!r.ok) return alert('save failed');
  V[String(clip().id)] = {...body}; status(); progress(); const i = D.clips.findIndex((c, k) => k > cur && !judged(c.id)); go(i >= 0 ? i : Math.min(cur + 1, D.clips.length - 1)); }
function stop(){ au.pause(); cancelAnimationFrame(raf); stopAt = null; }
function playRange(a, b){ stop(); au.src = `/wav/${clip().id}.wav`; const go2 = () => { au.currentTime = a; au.play(); stopAt = b;
  const tick = () => { draw(au.currentTime); if (au.paused || au.currentTime >= stopAt){ au.pause(); draw(null); return; } raf = requestAnimationFrame(tick); }; tick(); };
  au.readyState >= 1 && au.src.endsWith(`${clip().id}.wav`) ? go2() : (au.addEventListener('loadedmetadata', go2, {once: true}), au.load()); }
function playSyl(){ const s = clip().syllables[sel]; if (!s.win) return playRange(0, clip().dur); playRange(Math.max(0, s.win[0] - 0.3), Math.min(clip().dur, s.win[1] + 0.3)); }
let refAu = null; function ref(syl){ if (refAu) refAu.pause(); refAu = new Audio(`/ref/${syl}.mp3`); refAu.play().catch(() => {}); }
function draw(t){ const c = clip(), P = D.pitch && D.pitch[String(c.id)], dpr = devicePixelRatio, W = Math.round(cv.clientWidth * dpr), H = Math.round(260 * dpr);
  if (cv.width !== W) cv.width = W; if (cv.height !== H) cv.height = H; const ctx = cv.getContext('2d'), css = getComputedStyle(document.body), col = n => css.getPropertyValue(n);
  const gut = GUT * dpr, x = tt => gut + tt / c.dur * (W - gut), WH = 54 * dpr, top = WH + 8 * dpr, bot = H - 20 * dpr; ctx.clearRect(0, 0, W, H);
  c.syllables.forEach(s => { if (!s.win) return; ctx.globalAlpha = s.i === sel ? .5 : (s.i % 2 ? .10 : .20); ctx.fillStyle = s.i === sel ? col('--hl') : col('--mut');
    ctx.fillRect(x(s.win[0]), 0, x(s.win[1]) - x(s.win[0]), H - 16 * dpr); ctx.globalAlpha = 1; });
  ctx.fillStyle = col('--mut'); const nb = c.peaks.length, bw = (W - gut) / nb; c.peaks.forEach((p, i) => { const ph = Math.max(1, p / 100 * WH * .9); ctx.fillRect(gut + i * bw, (WH - ph) / 2, Math.max(1, bw - 1), ph); });
  ctx.font = `${11 * dpr}px system-ui`; ctx.textBaseline = 'middle';
  if (P && P.range && P.f0.some(v => v !== null)){ const voiced = P.f0.filter(v => v !== null), [lo, hi] = P.range;
    const ymin = Math.min(lo / 2 ** (3 / 12), Math.min(...voiced) / 2 ** (1 / 12)), ymax = Math.max(hi * 2 ** (3 / 12), Math.max(...voiced) * 2 ** (1 / 12));
    const y = hz => top + (bot - top) * (1 - Math.min(1, Math.max(0, Math.log2(hz / ymin) / Math.log2(ymax / ymin))));
    ctx.fillStyle = 'rgba(80,130,220,.13)'; ctx.fillRect(gut, y(hi), W - gut, y(lo) - y(hi)); ctx.textAlign = 'right'; ctx.lineWidth = dpr;
    for (const hz of [60, 75, 100, 125, 150, 200, 250, 300, 400, 500]) if (y(hz) > top + 6 * dpr && y(hz) < bot - 6 * dpr){ ctx.strokeStyle = col('--line');
      ctx.beginPath(); ctx.moveTo(gut, y(hz) + .5); ctx.lineTo(W, y(hz) + .5); ctx.stroke(); ctx.fillStyle = col('--mut'); ctx.fillText(String(hz), gut - 4 * dpr, y(hz)); }
    ctx.setLineDash([3 * dpr, 4 * dpr]); for (let l = 1; l <= 5; l++){ ctx.strokeStyle = 'rgba(80,130,220,.45)'; ctx.beginPath(); ctx.moveTo(gut, y(levelHz(l, lo, hi))); ctx.lineTo(W, y(levelHz(l, lo, hi))); ctx.stroke(); }
    ctx.setLineDash([]); ctx.lineCap = 'round'; ctx.lineWidth = 6 * dpr;
    c.syllables.forEach(s => { const tone = +s.label.slice(-1); if (!s.win || !CHAO[tone]) return; let a = Infinity, b = -Infinity;
      P.f0.forEach((v, k) => { const tt = P.t0 + k * P.step; if (v !== null && tt >= s.win[0] && tt <= s.win[1] + .02){ a = Math.min(a, tt); b = Math.max(b, tt); } });
      if (!(b - a > .04)){ a = s.win[0]; b = s.win[1]; } const [l0, l1] = CHAO[tone]; ctx.globalAlpha = .55; ctx.strokeStyle = TONE_COL[tone]; ctx.beginPath();
      for (let q = 0; q <= 8; q++){ const f = q / 8, px = x(a + (b - a) * f), py = y(levelHz(l0 + (l1 - l0) * f, lo, hi)); q ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke(); ctx.globalAlpha = 1; });
    ctx.fillStyle = col('--fg'); P.f0.forEach((v, k) => { if (v === null) return; ctx.beginPath(); ctx.arc(x(P.t0 + k * P.step), y(v), 2.2 * dpr, 0, 2 * Math.PI); ctx.fill(); }); }
  ctx.textAlign = 'center'; ctx.font = `bold ${12 * dpr}px system-ui`;
  c.syllables.forEach(s => { if (!s.win) return; ctx.fillStyle = marks[String(s.i)] !== undefined ? col('--bad') : col('--fg'); ctx.fillText(s.label, x((s.win[0] + s.win[1]) / 2), H - 8 * dpr); });
  if (t !== null && t !== undefined){ ctx.fillStyle = col('--acc'); ctx.fillRect(x(t), 0, 2 * dpr, H - 16 * dpr); } }
addEventListener('keydown', e => { if (e.ctrlKey || e.metaKey || e.altKey || (e.target && e.target.tagName === 'INPUT')) return; const k = e.key, n = clip().syllables.length;
  if (k === 'o') playRange(0, clip().dur); else if (k === 'p') playSyl(); else if (k === 'r') ref(clip().syllables[sel].label); else if (k === 'x') toggleWrong();
  else if (k === 'ArrowRight') select(Math.min(n - 1, sel + 1)); else if (k === 'ArrowLeft') select(Math.max(0, sel - 1));
  else if (k === '1') save('ok'); else if (k === '2') save('errors'); else if (k === '3') save('unusable'); else if (k === 'n') go(cur + 1); else if (k === 'b') go(cur - 1); });
init();
</script></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    b.add_argument("--decoded_train", default=os.path.join(OUT_DIR, "decode_full", "decoded_train.jsonl"))
    b.add_argument("--tojyutping_path", default=None)
    b.add_argument("--n", type=int, default=150)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out_dir", default=AUDIT_DIR)
    b.set_defaults(fn=cmd_build)
    p = sub.add_parser("pitch")
    p.add_argument("--out_dir", default=AUDIT_DIR)
    p.set_defaults(fn=cmd_pitch)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8778)
    s.add_argument("--out_dir", default=AUDIT_DIR)
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("summary")
    m.add_argument("--out_dir", default=AUDIT_DIR)
    m.add_argument("--label_dir", default=OUT_DIR)
    m.set_defaults(fn=cmd_summary)
    c = sub.add_parser("check")
    c.add_argument("--tojyutping_path", default=None)
    c.add_argument("--audit_dir", default=AUDIT_DIR)
    c.add_argument("--label_dir", default=OUT_DIR)
    c.add_argument("--out_dir", default=CHECK_DIR)
    c.set_defaults(fn=cmd_check)
    cs = sub.add_parser("check_summary")
    cs.add_argument("--out_dir", default=CHECK_DIR)
    cs.set_defaults(fn=cmd_check_summary)
    a = ap.parse_args()
    a.fn(a)
