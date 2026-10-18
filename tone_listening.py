"""Listening service for tone errors: is the label or the model right?  (blind A/B, verdicts saved on the server)

Tone-only substitutions of the 20k-clip decode (`mine_label_disagreements.py decode`, `group_errors.py`) are grouped by their
label->model syllable pair (e.g. wai6->wai4) and up to --per_group clips are sampled from every group.

    python tone_listening.py build  [--min_count 5] [--per_group 10]   # wavs + CTC syllable windows -> tmp/label_mining/tone/
        [--extra 度:dou3:dou6 --tojyutping_path <dir>]                   # + an agreement check: clips where label AND model say dou3
        [--fresh 度:dou3:dou6 --fresh_n 15]                             # + a rule check on clips OUTSIDE the 20k decode: label dou3 vs rule dou6
        [--fresh_model 時:si6:si4 --fresh_model_n 10]                   # + the same for a `model` rule: clips of the full decode the rule would change
    python tone_listening.py pitch                                      # F0 track + pitch range per clip (needs praat-parselmouth)
    python tone_listening.py serve  [--port 8777]                       # http://127.0.0.1:8777, verdicts -> tone/verdicts.jsonl
    python tone_listening.py summary                                    # who was right, per group and overall
    python tone_listening.py refs                                       # pre-fetch the reference syllables (the page also fetches on demand)

Every card also has a pitch panel like the playground's: F0 dots (Praat, 70-450 Hz, 10 ms), the clip's pitch range (10th-90th
percentile of voiced frames = Chao levels 1-5, dashed) and the canonical contours of both readings laid over the voiced part of
the syllable, so you can see which shape the F0 follows. `pitch` needs an interpreter with praat-parselmouth (the playground's
backend/.venv has it) and must be re-run after every `build`.

Clicking a reading (A / B) plays a reference recording of that isolated syllable: the mp3 that cantonesepinyin.com serves at
/api/audio/<syllable>, fetched once per syllable (robots.txt asks for 1 s between requests) and cached in tone/ref/ for local
use only. A single voice in citation tone: an anchor for the contour, not a verdict (running speech differs, mergers especially).

The page shows the two readings as A / B in a random order per clip, without saying which one is the label; the groups are
listed without direction until "reveal" is ticked. Syllable windows come from the CTC spikes of run21, so they are approximate:
"clip" always plays the whole recording.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import groupby

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import align  # noqa: E402
from mine_label_disagreements import OUT_DIR, convert  # noqa: E402

TONE_DIR = os.path.join(OUT_DIR, "tone")
VERDICTS = ("A", "B", "either", "neither")
REF_URL = "https://cantonesepinyin.com/api/audio/{}"
REF_DELAY = 1.2  # seconds between requests to the site (its robots.txt: Crawl-delay 1)
_ref_lock, _ref_last, _ref_missing = threading.Lock(), [0.0], set()


def fetch_ref(syllable, ref_dir):
    """path of the cached reference mp3 of one syllable, fetched from cantonesepinyin.com on first use (None if it has none)"""
    path = os.path.join(ref_dir, f"{syllable}.mp3")
    if os.path.exists(path):
        return path
    if not re.fullmatch(r"[a-z]+[1-6]", syllable) or syllable in _ref_missing:
        return None
    with _ref_lock:
        if os.path.exists(path):
            return path
        time.sleep(max(0.0, REF_DELAY - (time.time() - _ref_last[0])))
        _ref_last[0] = time.time()
        try:
            request = urllib.request.Request(REF_URL.format(syllable), headers={"User-Agent": "Mozilla/5.0 (tone-listening, local use)"})
            with urllib.request.urlopen(request, timeout=20) as r:
                data, ok = r.read(), r.headers.get_content_type() == "audio/mpeg"
        except (urllib.error.URLError, TimeoutError):
            data, ok = b"", False
        if not ok or len(data) < 500:
            _ref_missing.add(syllable)
            return None
        os.makedirs(ref_dir, exist_ok=True)
        open(path, "wb").write(data)
        return path


# ----------------------------------------------------------------------------- build
def frame_syllables(tokenizer, ids):
    """Greedy CTC ids -> [(syllable, first frame of the syllable, last frame of its rime token)], paired like jyutping.py does"""
    from jyutping import JYUTPING_RE, ONSETS, SPECIAL_TOKENS

    out, onset, pos = [], None, 0
    for key, run in groupby(ids):
        first, last = pos, pos + len(list(run)) - 1
        pos = last + 1
        if key == tokenizer.pad_token_id:
            continue
        token = tokenizer.convert_ids_to_tokens(key)
        if token == tokenizer.word_delimiter_token or token in SPECIAL_TOKENS:
            continue
        if token in ONSETS:
            if onset:
                out.append((onset[0], onset[1], onset[2]))
            onset = (token, first, last)
        elif JYUTPING_RE.match(token):
            out.append((onset[0] + token, onset[1], last) if onset else (token, first, last))
            onset = None
        else:
            if onset:
                out.append((onset[0], onset[1], onset[2]))
                onset = None
            out.append((token, first, last))
    if onset:
        out.append((onset[0], onset[1], onset[2]))
    return out


def pred_index(gold, pred, pos):
    """index in the prediction of the syllable aligned with gold[pos]"""
    j = 0
    for op, i, _, _ in align(gold, pred):
        if op in ("=", "S") and i == pos:
            return j
        if op in ("=", "S", "I"):
            j += 1
    return None


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

    grouped = [json.loads(line) for line in open(os.path.join(args.label_dir, "grouped_errors.jsonl"))]
    offsets = {(r["train_idx"], r["pos"]): r["offset"] for r in map(json.loads, open(os.path.join(args.label_dir, "substitutions.jsonl")))}
    decoded = {c["train_idx"]: c for c in map(json.loads, open(os.path.join(args.label_dir, "decoded.jsonl")))}
    tone = [r for r in grouped if r["op"] == "S" and r["parts"] == ["tone"]]
    by_pair = defaultdict(list)
    for r in tone:
        by_pair[(r["gold"], r["pred"])].append(r)
    keep = sorted(((k, v) for k, v in by_pair.items() if len(v) >= args.min_count), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(args.seed)
    picked = [(k, len(v), rng.sample(v, min(args.per_group, len(v)))) for k, v in keep]
    extras = []  # agreement checks: the label and the model say the same, is it right? (the alternative is the suspected reading)
    for spec in args.extra:
        char, reading, alt = spec.split(":")
        if args.tojyutping_path:
            sys.path.insert(0, args.tojyutping_path)
        import ToJyutping as tj

        cands = []
        for c in decoded.values():
            conv, chars, ok = convert(tj, c["text"])
            if not ok or len(conv) != len(c["gold"]):
                continue
            equal = {i for op, i, _, _ in align(c["gold"], c["pred"]) if op == "="}
            hits = [i for i, (ch, _) in enumerate(chars) if ch == char and i in equal and c["gold"][i] == reading]
            if hits:  # one syllable per clip
                cands.append({"train_idx": c["train_idx"], "pos": hits[0], "text": c["text"], "char": char, "conv": conv[hits[0]],
                              "offset": chars[hits[0]][1], "group": "agree_check"})
        extras.append(((reading, alt), len(cands), random.Random(f"{args.seed}-{spec}").sample(cands, min(args.per_group, len(cands)))))
    fresh = []  # rule checks on clips outside the 20k decode: is the label's old reading really wrong? (the rule's reading is the alternative)

    def take_fresh(candidates, tag, n):
        """up to n usable clips of (train index, position, text, label syllables), in a shuffle that depends only on the seed and the tag"""
        rows = []
        for idx, pos, text, syllables in random.Random(f"{args.seed}-{tag}").sample(candidates, len(candidates)):
            conv, chars, ok = convert(tj, text)
            if not ok or len(conv) != len(syllables):
                continue
            rows.append({"train_idx": idx, "pos": pos, "text": text, "char": chars[pos][0], "conv": conv[pos], "offset": chars[pos][1],
                         "group": "fresh_check"})
            decoded[idx] = {"train_idx": idx, "gold": syllables, "pred": None, "text": text}
            if len(rows) == n:
                break
        return rows

    if args.fresh:
        if args.tojyutping_path and args.tojyutping_path not in sys.path:
            sys.path.insert(0, args.tojyutping_path)
        import ToJyutping as tj
        from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
        from jyutping import normalize_jyutping_text
        from lexical_experiment import char_syllable_pairs

        specs = [tuple(s.split(":")) for s in args.fresh]
        pool = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select_columns(["text", "phone"])
        found = {spec: [] for spec in specs}
        for idx, (text, phone) in enumerate(zip(pool["text"], pool["phone"])):
            if idx in decoded:
                continue
            syllables = normalize_jyutping_text(phone).split()
            pairs = char_syllable_pairs(tj, text, syllables)
            if pairs is None:
                continue
            for spec in specs:
                hits = [i for i, pair in enumerate(pairs) if pair == spec[:2]]
                if hits:
                    found[spec].append((idx, hits[0], text, syllables))
        for spec in specs:
            fresh.append(((spec[1], spec[2]), len(found[spec]), take_fresh(found[spec], f"fresh-{':'.join(spec)}", args.fresh_n)))
    if args.fresh_model:
        # rules that apply only where the model says the corrected reading: sample the clips such a rule would change, taken from the
        # full decode of the training clips (decode_train.py), outside the 20k decode
        if args.tojyutping_path and args.tojyutping_path not in sys.path:
            sys.path.insert(0, args.tojyutping_path)
        import ToJyutping as tj
        from lexical_experiment import char_syllable_pairs

        specs = [tuple(s.split(":")) for s in args.fresh_model]
        found = {spec: [] for spec in specs}
        for line in open(args.decoded_train):
            clip = json.loads(line)
            if clip["train_idx"] in decoded:
                continue
            pairs = char_syllable_pairs(tj, clip["text"], clip["gold"])
            if pairs is None:
                continue
            model_at = None
            for spec in specs:
                for i, pair in enumerate(pairs):
                    if pair == spec[:2]:
                        model_at = model_at or {k: (g if op == "=" else p) for op, k, g, p in align(clip["gold"], clip["pred"]) if k is not None}
                        if model_at.get(i) == spec[2]:
                            found[spec].append((clip["train_idx"], i, clip["text"], clip["gold"]))
                        break
        for spec in specs:
            fresh.append(((spec[1], spec[2]), len(found[spec]), take_fresh(found[spec], f"freshm-{':'.join(spec)}", args.fresh_model_n)))
    clips = sorted({r["train_idx"] for _, _, rs in picked + extras + fresh for r in rs})
    print(f"{len(tone)} tone-only substitutions, {len(by_pair)} label->model pairs, {len(keep)} with >= {args.min_count}; "
          f"{sum(len(rs) for _, _, rs in picked)} samples from {len(clips)} clips", flush=True)

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

    info = {}
    for k, idx in enumerate(clips):
        row = train.select([idx]).cast_column("audio", Audio(decode=False))[0]
        y, sr = read_audio(row["audio"])
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        y = y.astype(np.float32)
        sf.write(os.path.join(wav_dir, f"{idx}.wav"), y, 16000, subtype="PCM_16")
        inp = proc(y, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(input_features=inp.input_features.to(dev), attention_mask=inp.attention_mask.to(dev), return_dict=True).logits
        n = int(model._get_feat_extract_output_lengths(inp.attention_mask.sum(-1))[0])
        syls = frame_syllables(tok, logits[0].argmax(-1).tolist()[:n])
        n_bins = 240
        edge = len(y) // n_bins * n_bins
        peaks = np.abs(y[:edge]).reshape(n_bins, -1).max(1)
        info[idx] = {"dur": len(y) / 16000, "frames": n, "syls": syls,
                     "peaks": [int(round(100 * p / max(float(peaks.max()), 1e-6))) for p in peaks]}
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(clips)} clips", flush=True)

    for _, _, rs in fresh:
        for r in rs:
            decoded[r["train_idx"]]["pred"] = [s for s, _, _ in info[r["train_idx"]]["syls"]]

    def window(idx, sample):
        c, i = decoded[idx], info[idx]
        j = pred_index(c["gold"], c["pred"], sample["pos"])
        if j is None or [s for s, _, _ in i["syls"]] != c["pred"]:
            return None, None
        # as in ../wav2vec2-jyutping-playground (alignment.syllable_intervals): a syllable ends at the end of the frame of its
        # rime spike (frame k is centred at 0.03 + 0.04 k s) and starts where the previous one ended, the first 0.25 s earlier
        t = lambda f: 0.05 + 0.04 * f
        end = min(i["dur"], t(i["syls"][j][2]))
        start = max(0.0, end - 0.25) if j == 0 else min(i["dur"], t(i["syls"][j - 1][2]))
        start = max(start, end - 0.6)  # a long pause before the syllable is not part of it
        end = min(max(end, start + 0.05), i["dur"])
        return [round(start, 3), round(end, 3)], [round(max(0.0, start - 0.3), 3), round(min(i["dur"], end + 0.3), 3)]

    groups, no_window = [], 0
    for kind, (gold, pred), n_total, rs in [("agree", *e) for e in extras] + [("fresh", *f) for f in fresh] + [(None, *p) for p in picked]:
        samples = []
        for r in rs:
            shade, seg = window(r["train_idx"], r)
            no_window += shade is None
            samples.append({
                "id": f"{r['train_idx']}_{r['pos']}", "clip": r["train_idx"], "text": r["text"], "char": r["char"],
                "offset": r["offset"] if "offset" in r else offsets[(r["train_idx"], r["pos"])], "gold": gold, "pred": pred, "conv": r["conv"], "taxonomy": r["group"],
                "a_is_label": random.Random(f"{r['train_idx']}-{r['pos']}").random() < 0.5, "dur": info[r["train_idx"]]["dur"],
                "shade": shade, "seg": seg, "peaks": info[r["train_idx"]]["peaks"], **({"kind": kind} if kind else {})})
        groups.append({"gold": gold, "pred": pred, "n_total": n_total, "samples": samples, **({"kind": kind} if kind else {})})
    json.dump({"model": args.model_id, "seed": args.seed, "clips": len(clips), "groups": groups},
              open(os.path.join(args.out_dir, "samples.json"), "w"), ensure_ascii=False)
    n_samples = sum(len(g["samples"]) for g in groups)
    print(f"wrote {n_samples} samples in {len(groups)} groups ({no_window} without a syllable window) -> {args.out_dir}", flush=True)


def cmd_pitch(args):
    """Praat F0 track (the playground's settings) and the 10th-90th percentile pitch range of every clip, into samples.json"""
    import wave

    import numpy as np
    import parselmouth

    path = os.path.join(args.out_dir, "samples.json")
    data = json.load(open(path))
    pitch = {}
    for clip in sorted({s["clip"] for g in data["groups"] for s in g["samples"]}):
        with wave.open(os.path.join(args.out_dir, "wav", f"{clip}.wav")) as w:
            rate, frames = w.getframerate(), w.readframes(w.getnframes())
        y = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768
        track = parselmouth.Sound(y, sampling_frequency=rate).to_pitch(time_step=0.01, pitch_floor=70.0, pitch_ceiling=450.0)
        f0 = track.selected_array["frequency"]
        voiced = f0[f0 > 0]
        rng = None
        if len(voiced):
            low, high = (float(v) for v in np.percentile(voiced, [10, 90]))
            if high / low < 2 ** (4 / 12):  # at least four semitones wide
                centre, half = (low * high) ** 0.5, 2 ** (4 / 24)
                low, high = centre / half, centre * half
            rng = [round(low, 1), round(high, 1)]
        pitch[str(clip)] = {"t0": round(float(track.xs()[0]), 4), "step": 0.01, "range": rng,
                            "f0": [round(float(v), 1) if v > 0 else None for v in f0]}
    data["pitch"] = pitch
    json.dump(data, open(path, "w"), ensure_ascii=False)
    print(f"pitch track for {len(pitch)} clips ({sum(p['range'] is None for p in pitch.values())} without voiced frames) -> {path}")


# ----------------------------------------------------------------------------- serve
class Handler(BaseHTTPRequestHandler):
    tone_dir = TONE_DIR
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, status, body, ctype, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _verdicts(self):
        out, path = {}, os.path.join(self.tone_dir, "verdicts.jsonl")
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                out[r["id"]] = r
        return out

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/data":
            return self._send(200, open(os.path.join(self.tone_dir, "samples.json"), "rb").read(), "application/json")
        if path == "/api/verdicts":
            return self._send(200, json.dumps(self._verdicts(), ensure_ascii=False).encode(), "application/json")
        m = re.fullmatch(r"/wav/(\d+)\.wav", path)
        if m and os.path.exists(os.path.join(self.tone_dir, "wav", f"{m.group(1)}.wav")):
            return self._send_audio(os.path.join(self.tone_dir, "wav", f"{m.group(1)}.wav"), "audio/wav")
        m = re.fullmatch(r"/ref/([a-z]+[1-6])\.mp3", path)
        if m:
            ref = fetch_ref(m.group(1), os.path.join(self.tone_dir, "ref"))
            if ref:
                return self._send_audio(ref, "audio/mpeg")
        self._send(404, b"not found", "text/plain")

    def _send_audio(self, path, ctype):
        data = open(path, "rb").read()
        size, status, start, end, extra = len(data), 200, 0, len(data) - 1, {"Accept-Ranges": "bytes"}
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                start = int(rng.group(1))
                end = int(rng.group(2)) if rng.group(2) else size - 1
            else:
                start = max(size - int(rng.group(2)), 0)
            end = min(end, size - 1)
            if start > end:
                return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
            status, extra["Content-Range"] = 206, f"bytes {start}-{end}/{size}"
        self._send(status, data[start:end + 1], ctype, extra)

    def do_POST(self):
        if self.path != "/api/verdict":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            assert isinstance(body.get("id"), str) and re.fullmatch(r"\d+_\d+", body["id"]) and body.get("verdict") in VERDICTS
        except Exception:
            return self._send(400, b"bad request", "text/plain")
        with self.lock, open(os.path.join(self.tone_dir, "verdicts.jsonl"), "a") as f:
            f.write(json.dumps({"id": body["id"], "verdict": body["verdict"], "ts": time.time()}) + "\n")
        self._send(200, b"{}", "application/json")


def cmd_serve(args):
    Handler.tone_dir = args.out_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {args.out_dir} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


def cmd_refs(args):
    data = json.load(open(os.path.join(args.out_dir, "samples.json")))
    syllables = sorted({s for g in data["groups"] for s in (g["gold"], g["pred"])})
    cached = sum(os.path.exists(os.path.join(args.out_dir, "ref", f"{s}.mp3")) for s in syllables)
    print(f"{len(syllables)} reference syllables, {cached} already cached; fetching the rest at one request per {REF_DELAY} s", flush=True)
    missing = [s for s in syllables if not fetch_ref(s, os.path.join(args.out_dir, "ref"))]
    print(f"done; the site has no recording for {len(missing)}: {' '.join(missing) or '-'}")


# ----------------------------------------------------------------------------- summary
def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_summary(args):
    data = json.load(open(os.path.join(args.out_dir, "samples.json")))
    verdicts = {}
    path = os.path.join(args.out_dir, "verdicts.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            verdicts[r["id"]] = r["verdict"]

    def who(s, v):
        if v in ("either", "neither"):
            return v
        return "label" if (v == "A") == s["a_is_label"] else "model"

    total, tax = Counter(), defaultdict(Counter)
    print(f"{'label->model':16} {'n':>3} {'label':>6} {'model':>6} {'either':>7} {'neither':>8}   model right among label/model")
    n_all = n_judged = 0
    agree_groups = [g for g in data["groups"] if g.get("kind") == "agree"]
    fresh_groups = [g for g in data["groups"] if g.get("kind") == "fresh"]
    data["groups"] = [g for g in data["groups"] if not g.get("kind")]
    for g in data["groups"]:
        c = Counter()
        for s in g["samples"]:
            n_all += 1
            if s["id"] in verdicts:
                n_judged += 1
                w = who(s, verdicts[s["id"]])
                c[w] += 1
                tax[s["taxonomy"].split("/")[0]][w] += 1
        total.update(c)
        if sum(c.values()):
            dec = c["label"] + c["model"]
            share = f"{c['model'] / dec:.0%} ({c['model']}/{dec})" if dec else "-"
            print(f"{g['gold'] + '->' + g['pred']:16} {sum(c.values()):3d} {c['label']:6d} {c['model']:6d} {c['either']:7d} {c['neither']:8d}   {share}")
    dec = total["label"] + total["model"]
    lo, hi = wilson(total["model"], dec)
    print(f"\njudged {n_judged} of {n_all} samples; label {total['label']}, model {total['model']}, either {total['either']}, neither {total['neither']}")
    if dec:
        print(f"model right in {total['model'] / dec:.1%} of the {dec} clips where the listener picked label or model (95% CI {lo:.0%}-{hi:.0%})")
    for t, c in sorted(tax.items()):
        d = c["label"] + c["model"]
        if d:
            print(f"  {t:16} model right {c['model']}/{d} = {c['model'] / d:.0%}   (either {c['either']}, neither {c['neither']})")
    for g in agree_groups:
        c = Counter(who(s, verdicts[s["id"]]) for s in g["samples"] if s["id"] in verdicts)
        if sum(c.values()):
            print(f"\nagreement check, label = model = {g['gold']} ({g['n_total']} such syllables in the 20k clips), alternative {g['pred']}: "
                  f"{g['gold']} right {c['label']}, {g['pred']} right {c['model']}, either {c['either']}, neither {c['neither']} (of {sum(c.values())} judged)")
    for g in fresh_groups:
        c = Counter(who(s, verdicts[s["id"]]) for s in g["samples"] if s["id"] in verdicts)
        if sum(c.values()):
            print(f"\nfresh rule check, {g['samples'][0]['char']} label {g['gold']} vs rule {g['pred']} ({g['n_total']} such labels outside the 20k clips): "
                  f"label right {c['label']}, rule right {c['model']}, either {c['either']}, neither {c['neither']} (of {sum(c.values())} judged)")


PAGE = r"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Tone listening</title>
<style>
:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--card:#f4f4f6;--line:#d9d9de;--acc:#2563eb;--ok:#15803d;--hl:#fde68a}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa3;--card:#1f1f23;--line:#33333a;--acc:#6ea0ff;--ok:#4ade80;--hl:#6b5a14}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
header b{font-size:16px}.mut{color:var(--mut)}
#wrap{display:flex;gap:16px;padding:16px;align-items:flex-start}
#side{width:210px;flex:none;position:sticky;top:60px;max-height:calc(100vh - 80px);overflow:auto}
#side div{padding:6px 10px;border-radius:8px;cursor:pointer;display:flex;justify-content:space-between;gap:8px}
#side div:hover{background:var(--card)}#side .on{background:var(--card);outline:2px solid var(--acc)}#side .done{color:var(--ok)}
#main{flex:1;min-width:0}
.card{background:var(--card);border-radius:12px;padding:12px 14px;margin-bottom:12px;border:2px solid transparent}.card.act{border-color:var(--acc)}
.txt{font-size:22px;margin-bottom:6px}mark{background:var(--hl);color:inherit;border-radius:4px;padding:0 2px}
canvas{width:100%;height:56px;display:block;background:var(--bg);border-radius:8px;cursor:pointer}
canvas.pitch{height:170px;margin-top:6px;cursor:crosshair}.rdout{font:12px ui-monospace,monospace;color:var(--mut);min-height:16px;margin-top:2px}
.chip{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 3px 0 6px;vertical-align:-1px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;align-items:center}
button{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.sel{background:var(--acc);border-color:var(--acc);color:#fff}
button.rd{font-size:20px;font-weight:600;letter-spacing:.5px;padding:4px 14px}button.rd.missing{text-decoration:line-through;opacity:.5}button.rd.on{border-color:var(--acc);color:var(--acc)}
.info{font-size:13px;color:var(--mut);margin-top:6px}
.legend{font-size:13px;color:var(--mut)}
@media(max-width:700px){#wrap{flex-direction:column}#side{width:100%;position:static;max-height:200px}}
</style>
<header><b>Tone listening</b><span id=prog class=mut></span>
<label><input type=checkbox id=rev> reveal label / model</label>
<label><input type=checkbox id=zoom checked> pitch panel zoomed to the syllable</label>
<span class=legend>tones: 1 high level 55 · 2 high rising 25 · 3 mid level 33 · 4 low falling 21 · 5 low rising 23 · 6 low level 22</span>
<span class=legend>click a reading to hear a reference voice (isolated syllable, cantonesepinyin.com)</span>
<span class=legend>pitch panel: dots = F0 · blue band and dashed lines 1-5 = this clip's pitch range (Chao levels) · thick lines = the canonical contour of A and B over the voiced part of the syllable
<span id=chips></span></span>
<span class=legend>keys: o clip · p syllable · a/b/c reference A / B / A then B · 1 A · 2 B · 3 either · 4 neither · j/k next/prev</span></header>
<div id=wrap><nav id=side></nav><section id=main></section></div>
<script>
let D, V = {}, gi = 0, act = 0, raf = 0, cur = null;
const $ = s => document.querySelector(s), reveal = () => $('#rev').checked;
const NAMES = {A:'A', B:'B', either:'either / can\'t tell', neither:'neither / bad audio'};
async function init(){ D = (await (await fetch('/api/data')).json()); V = await (await fetch('/api/verdicts')).json();
  const seen = {}; [...D.groups.filter(g => !g.kind), ...D.groups.filter(g => g.kind)].forEach(g => { g.pair = [g.gold, g.pred].sort().join(' / '); seen[g.pair] = (seen[g.pair]||0)+1; g.ord = seen[g.pair]; });
  D.groups.forEach(g => g.dup = seen[g.pair]); $('#rev').onchange = () => { renderSide(); renderMain(); };
  $('#zoom').onchange = redrawPitch; addEventListener('resize', redrawPitch);
  $('#chips').innerHTML = [1,2,3,4,5,6].map(t => `<i class=chip style="background:${TONE_COL[t]}"></i>T${t}`).join(' ');
  renderSide(); renderMain(); }
const redrawPitch = () => document.querySelectorAll('.card').forEach(c => { drawWave(c.querySelector('canvas'), c._s, null); drawPitch(c, null); });
const MARK = {agree: '★ ', fresh: '☆ ', audit: '◇ '};
const gtitle = g => (MARK[g.kind] || '') + (reveal() ? (g.kind === 'agree' ? `${g.gold} (label = model) vs ${g.pred}` : g.kind === 'fresh' ? `${g.gold} (label) vs ${g.pred} (rule)` : `${g.gold} → ${g.pred}`) : g.pair + (g.dup > 1 ? ` #${g.ord}` : ''));
function renderSide(){ const side = $('#side'); side.innerHTML = ''; let all = 0, doneAll = 0;
  D.groups.forEach((g, i) => { const n = g.samples.length, d = g.samples.filter(s => V[s.id]).length; all += n; doneAll += d;
    const el = document.createElement('div'); el.className = (i === gi ? 'on ' : '') + (d === n ? 'done' : '');
    el.innerHTML = `<span>${gtitle(g)}</span><span class=mut>${d}/${n} · ${g.n_total}</span>`; el.onclick = () => { gi = i; act = 0; renderSide(); renderMain(); }; side.appendChild(el); });
  $('#prog').textContent = `judged ${doneAll} / ${all}`; }
function drawWave(cv, s, t){ const w = cv.width = cv.clientWidth * devicePixelRatio, h = cv.height = 56 * devicePixelRatio, ctx = cv.getContext('2d');
  const css = getComputedStyle(document.body); ctx.clearRect(0,0,w,h);
  if (s.shade){ ctx.fillStyle = css.getPropertyValue('--hl'); ctx.fillRect(s.shade[0]/s.dur*w, 0, (s.shade[1]-s.shade[0])/s.dur*w, h); }
  ctx.fillStyle = css.getPropertyValue('--mut'); const n = s.peaks.length, bw = w / n;
  s.peaks.forEach((p, i) => { const ph = Math.max(1, p/100*h*.9); ctx.fillRect(i*bw, (h-ph)/2, Math.max(1, bw-1), ph); });
  if (t != null){ ctx.fillStyle = css.getPropertyValue('--acc'); ctx.fillRect(t/s.dur*w, 0, 2*devicePixelRatio, h); } }
let refAu = null;
function stopAll(){ document.querySelectorAll('audio').forEach(a => a.pause()); cancelAnimationFrame(raf); if (refAu){ refAu.onended = null; refAu.pause(); } document.querySelectorAll('button.rd.on').forEach(b => b.classList.remove('on')); }
function playRef(card, which){ stopAll(); const btns = [...card.querySelectorAll('button.rd')], seq = which === 'ab' ? btns : [btns[which === 'a' ? 0 : 1]]; let i = 0;
  const next = () => { if (i >= seq.length) return; const bn = seq[i++], au = refAu = new Audio(`/ref/${bn.dataset.ref}.mp3`); bn.classList.add('on');
    au.onended = () => { bn.classList.remove('on'); setTimeout(next, 300); }; au.onerror = () => { bn.classList.remove('on'); bn.classList.add('missing'); bn.title = 'no reference recording'; };
    au.play().catch(() => {}); }; next(); }
function playRange(card, a, b){ const s = card._s, au = card.querySelector('audio'), cv = card.querySelector('canvas'); stopAll(); cur = card;
  const go = () => { au.currentTime = a; au.play(); const tick = () => { drawWave(cv, s, au.currentTime); drawPitch(card, au.currentTime);
      if (au.paused || au.currentTime >= b){ au.pause(); drawWave(cv, s, null); drawPitch(card, null); return; } raf = requestAnimationFrame(tick); }; tick(); };
  if (au.readyState >= 1) go(); else { au.addEventListener('loadedmetadata', go, {once:true}); au.load(); } }
function play(card, seg){ const s = card._s; playRange(card, seg && s.seg ? s.seg[0] : 0, seg && s.seg ? s.seg[1] : s.dur); }
// pitch panel, after the playground: Praat F0 dots, the clip's pitch range (10th-90th percentile) as Chao levels 1-5, and the
// canonical contour of each reading laid over the voiced part of the syllable window (levels are even in semitones)
const TONE_COL = {1:'#d55e00', 2:'#e69f00', 3:'#009e73', 4:'#0072b2', 5:'#cc79a7', 6:'#56b4e9'};
const CHAO = {1:[5,5], 2:[2,5], 3:[3,3], 4:[2,1], 5:[2,3], 6:[2,2]}, GUT = 38;
const levelHz = (l, lo, hi) => lo * Math.pow(hi / lo, (l - 1) / 4);
function pitchGeom(card){ const s = card._s, P = D.pitch && D.pitch[s.clip]; if (!P) return null;
  const [t0, t1] = $('#zoom').checked && s.seg ? s.seg : [0, s.dur]; return {s, P, t0, t1}; }
function drawPitch(card, t){ const cv = card.querySelector('canvas.pitch'), g = pitchGeom(card); if (!g){ cv.style.display = 'none'; return; }
  const {s, P, t0, t1} = g, dpr = devicePixelRatio, W = Math.round(cv.clientWidth * dpr), H = Math.round(170 * dpr);
  if (cv.width !== W) cv.width = W; if (cv.height !== H) cv.height = H;
  const ctx = cv.getContext('2d'), css = getComputedStyle(document.body), col = n => css.getPropertyValue(n), gut = GUT * dpr;
  const x = tt => gut + (tt - t0) / (t1 - t0) * (W - gut), voiced = P.f0.filter(v => v !== null);
  ctx.clearRect(0, 0, W, H); ctx.font = `${11 * dpr}px system-ui`; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
  if (!P.range || !voiced.length){ ctx.fillStyle = col('--mut'); ctx.fillText('no voiced frames in this clip', gut + 8 * dpr, H / 2); return; }
  const [lo, hi] = P.range, ymin = Math.min(lo / 2 ** (3 / 12), Math.min(...voiced) / 2 ** (1 / 12)), ymax = Math.max(hi * 2 ** (3 / 12), Math.max(...voiced) * 2 ** (1 / 12));
  const y = hz => H * (1 - Math.min(1, Math.max(0, Math.log2(hz / ymin) / Math.log2(ymax / ymin))));
  if (s.shade){ ctx.globalAlpha = .45; ctx.fillStyle = col('--hl'); ctx.fillRect(x(s.shade[0]), 0, x(s.shade[1]) - x(s.shade[0]), H); ctx.globalAlpha = 1; }
  ctx.fillStyle = 'rgba(80,130,220,.13)'; ctx.fillRect(gut, y(hi), W - gut, y(lo) - y(hi));
  ctx.lineWidth = dpr; ctx.textAlign = 'right';
  for (const hz of [60, 75, 100, 125, 150, 200, 250, 300, 400, 500]) if (y(hz) > 9 * dpr && y(hz) < H - 16 * dpr){
    ctx.strokeStyle = col('--line'); ctx.beginPath(); ctx.moveTo(gut, y(hz) + .5); ctx.lineTo(W, y(hz) + .5); ctx.stroke(); ctx.fillStyle = col('--mut'); ctx.fillText(String(hz), gut - 4 * dpr, y(hz)); }
  ctx.save(); ctx.beginPath(); ctx.rect(gut, 0, W - gut, H); ctx.clip(); ctx.setLineDash([3 * dpr, 4 * dpr]);
  for (let l = 1; l <= 5; l++){ const yy = y(levelHz(l, lo, hi)); ctx.strokeStyle = 'rgba(80,130,220,.5)'; ctx.beginPath(); ctx.moveTo(gut, yy); ctx.lineTo(W, yy); ctx.stroke();
    ctx.fillStyle = 'rgba(80,130,220,.95)'; ctx.fillText(String(l), W - 4 * dpr, yy - 6 * dpr); }
  ctx.setLineDash([]);
  const [w0, w1] = s.shade || [0, s.dur]; let a = Infinity, b = -Infinity;
  P.f0.forEach((v, k) => { const tt = P.t0 + k * P.step; if (v !== null && tt >= w0 && tt <= w1 + .02){ a = Math.min(a, tt); b = Math.max(b, tt); } });
  if (!(b - a > .04)){ a = w0; b = w1; }
  const labels = []; ctx.lineCap = 'round'; ctx.lineWidth = 6 * dpr; ctx.globalAlpha = .55;
  [['A', s.a_is_label ? s.gold : s.pred], ['B', s.a_is_label ? s.pred : s.gold]].forEach(([name, syl]) => { const tone = +syl.slice(-1), [l0, l1] = CHAO[tone];
    ctx.strokeStyle = TONE_COL[tone]; ctx.beginPath();
    for (let q = 0; q <= 8; q++){ const f = q / 8; q ? ctx.lineTo(x(a + (b - a) * f), y(levelHz(l0 + (l1 - l0) * f, lo, hi))) : ctx.moveTo(x(a + (b - a) * f), y(levelHz(l0 + (l1 - l0) * f, lo, hi))); }
    ctx.stroke(); labels.push([`${name} · T${tone}`, TONE_COL[tone], y(levelHz(l1, lo, hi))]); });
  ctx.globalAlpha = 1; ctx.fillStyle = col('--fg');
  P.f0.forEach((v, k) => { const tt = P.t0 + k * P.step; if (v === null || tt < t0 || tt > t1) return; ctx.beginPath(); ctx.arc(x(tt), y(v), 2.2 * dpr, 0, 2 * Math.PI); ctx.fill(); });
  const right = x(b) + 60 * dpr < W; ctx.font = `bold ${12 * dpr}px system-ui`; ctx.textAlign = right ? 'left' : 'right';
  labels.forEach(([txt, c, yy]) => { ctx.fillStyle = c; ctx.fillText(txt, right ? x(b) + 8 * dpr : x(a) - 8 * dpr, yy); });
  if (t !== null && t !== undefined && t >= t0 && t <= t1){ ctx.fillStyle = col('--acc'); ctx.fillRect(x(t), 0, 2 * dpr, H); }
  ctx.restore();
  ctx.fillStyle = col('--mut'); ctx.font = `${10 * dpr}px system-ui`; ctx.textAlign = 'center'; const step = t1 - t0 > 1.2 ? .5 : .1;
  for (let tt = Math.ceil(t0 / step) * step; tt < t1; tt += step) ctx.fillText(tt.toFixed(1), x(tt), H - 7 * dpr); }
function pitchReadout(card, e){ const g = pitchGeom(card), out = card.querySelector('.rdout'); if (!g) return; const r = e.target.getBoundingClientRect();
  const tt = g.t0 + ((e.clientX - r.left) - GUT) / (r.width - GUT) * (g.t1 - g.t0); if (tt < g.t0 || tt > g.t1){ out.textContent = ''; return; }
  const v = g.P.f0[Math.round((tt - g.P.t0) / g.P.step)], lo = g.P.range && g.P.range[0];
  out.textContent = `${tt.toFixed(3)} s · ` + (v ? `${Math.round(v)} Hz` + (lo ? ` · ${12 * Math.log2(v / lo) >= 0 ? '+' : '-'}${Math.abs(12 * Math.log2(v / lo)).toFixed(1)} st above level 1` : '') : 'unvoiced'); }
function pitchTime(card, e){ const g = pitchGeom(card), r = e.target.getBoundingClientRect(); return Math.min(g.t1, Math.max(g.t0, g.t0 + ((e.clientX - r.left) - GUT) / (r.width - GUT) * (g.t1 - g.t0))); }
function renderMain(){ const g = D.groups[gi], main = $('#main'); main.innerHTML = '';
  g.samples.forEach((s, k) => { const a = s.a_is_label ? s.gold : s.pred, b = s.a_is_label ? s.pred : s.gold;
    const card = document.createElement('div'); card.className = 'card' + (k === act ? ' act' : ''); card._s = s; card.id = 'c' + k;
    const t = s.text, o = s.offset, hasMark = s.char && t.substr(o, s.char.length) === s.char;
    card.innerHTML = `<div class=txt></div><canvas></canvas><canvas class=pitch></canvas><div class=rdout></div><audio preload=none src="/wav/${s.clip}.wav"></audio>
      <div class=row><button data-p=clip>▶ clip (o)</button><button data-p=seg ${s.seg ? '' : 'disabled'}>▶ syllable ± context (p)</button>
      <button class=rd data-ref="${a}" title="play reference voice">▶ ${a}</button><span class=mut>or</span><button class=rd data-ref="${b}" title="play reference voice">▶ ${b}</button>
      <button data-r=ab title="reference A then B (c)">A → B</button>
      <span style="flex:1"></span>${['A','B','either','neither'].map((v, i) => `<button data-v=${v}>${i+1} ${NAMES[v]}</button>`).join('')}</div>
      <div class=info></div>`;
    const tx = card.querySelector('.txt'); if (hasMark){ tx.append(t.slice(0, o)); const m = document.createElement('mark'); m.textContent = s.char; tx.append(m, t.slice(o + s.char.length)); } else tx.textContent = t;
    card.querySelector('.info').textContent = reveal() && s.kind === 'audit' ? `label ${s.gold} · alternative ${s.pred} · ${s.note}` : reveal() && s.kind === 'fresh' ? `label ${s.gold} · rule says ${s.pred} · ToJyutping ${s.conv || '-'} · A is the ${s.a_is_label ? 'label' : 'rule reading'}` : reveal() && s.kind ? `label = model = ${s.gold} · alternative ${s.pred} · ToJyutping ${s.conv || '-'} · A is the ${s.a_is_label ? 'label/model reading' : 'alternative'}` : reveal() ? `label ${s.gold} · model ${s.pred} · ToJyutping ${s.conv || '-'} · ${s.taxonomy} · A is the ${s.a_is_label ? 'label' : 'model'}` : `${s.dur.toFixed(1)} s · shaded = approximate syllable position`;
    card.querySelectorAll('[data-p]').forEach(bn => bn.onclick = () => { setAct(k); play(card, bn.dataset.p === 'seg'); });
    card.querySelectorAll('[data-v]').forEach(bn => bn.onclick = () => { setAct(k); verdict(k, bn.dataset.v); });
    card.querySelectorAll('button.rd').forEach((bn, i) => bn.onclick = () => { setAct(k); playRef(card, i ? 'b' : 'a'); });
    card.querySelector('[data-r=ab]').onclick = () => { setAct(k); playRef(card, 'ab'); };
    card.querySelector('canvas').onclick = e => { const r = e.target.getBoundingClientRect(); setAct(k); playRange(card, (e.clientX - r.left) / r.width * s.dur, s.dur); };
    const cp = card.querySelector('canvas.pitch'); cp.onmousemove = e => pitchReadout(card, e); cp.onmouseleave = () => { card.querySelector('.rdout').textContent = ''; };
    cp.onclick = e => { if (!pitchGeom(card)) return; setAct(k); playRange(card, pitchTime(card, e), s.dur); };
    main.appendChild(card); mark(card); drawWave(card.querySelector('canvas'), s, null); drawPitch(card, null); }); }
function mark(card){ const v = V[card._s.id]; card.querySelectorAll('[data-v]').forEach(bn => bn.classList.toggle('sel', !!v && bn.dataset.v === v.verdict)); }
function setAct(k){ act = k; document.querySelectorAll('.card').forEach((c, i) => c.classList.toggle('act', i === k)); }
async function verdict(k, v){ const card = document.getElementById('c' + k), s = card._s;
  const r = await fetch('/api/verdict', {method:'POST', body: JSON.stringify({id: s.id, verdict: v})}); if (!r.ok) return alert('save failed');
  V[s.id] = {verdict: v}; mark(card); renderSide(); }
addEventListener('keydown', e => { if (e.ctrlKey || e.metaKey || e.altKey) return; const n = D.groups[gi].samples.length, card = document.getElementById('c' + act);
  if (e.key === 'j'){ setAct(Math.min(n-1, act+1)); document.getElementById('c' + act).scrollIntoView({block:'center'}); }
  else if (e.key === 'k'){ setAct(Math.max(0, act-1)); document.getElementById('c' + act).scrollIntoView({block:'center'}); }
  else if (e.key === 'o') play(card, false); else if (e.key === 'p') play(card, true);
  else if (e.key === 'a') playRef(card, 'a'); else if (e.key === 'b') playRef(card, 'b'); else if (e.key === 'c') playRef(card, 'ab');
  else if ('1234'.includes(e.key) && e.key) verdict(act, VERDICTS_[+e.key - 1]); });
const VERDICTS_ = ['A','B','either','neither'];
init();
</script></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    b.add_argument("--min_count", type=int, default=5, help="smallest label->model group (in the 20k decode) that is listed")
    b.add_argument("--per_group", type=int, default=10)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--extra", action="append", default=[], metavar="CHAR:READING:ALT",
                   help="agreement check: clips where the label and the model both say READING for CHAR; ALT is the suspected reading")
    b.add_argument("--fresh", action="append", default=[], metavar="CHAR:LABEL:RULE",
                   help="rule check on clips outside the 20k decode whose label is LABEL for CHAR; RULE is the corrected reading")
    b.add_argument("--fresh_n", type=int, default=15, help="clips per --fresh check")
    b.add_argument("--fresh_model", action="append", default=[], metavar="CHAR:LABEL:RULE",
                   help="like --fresh for a rule that applies only where the model says RULE: clips of the full decode with that label and that model syllable")
    b.add_argument("--fresh_model_n", type=int, default=10, help="clips per --fresh_model check")
    b.add_argument("--decoded_train", default=os.path.join(OUT_DIR, "decode_full", "decoded_train.jsonl"),
                   help="full decode of the training clips (decode_train.py), for --fresh_model")
    b.add_argument("--tojyutping_path", default=None, help="needed for --extra and --fresh (see check_labels_vs_tojyutping.py)")
    b.add_argument("--label_dir", default=OUT_DIR)
    b.add_argument("--out_dir", default=TONE_DIR)
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8777)
    s.add_argument("--out_dir", default=TONE_DIR)
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("summary")
    m.add_argument("--out_dir", default=TONE_DIR)
    m.set_defaults(fn=cmd_summary)
    r = sub.add_parser("refs")
    r.add_argument("--out_dir", default=TONE_DIR)
    r.set_defaults(fn=cmd_refs)
    p = sub.add_parser("pitch")
    p.add_argument("--out_dir", default=TONE_DIR)
    p.set_defaults(fn=cmd_pitch)
    a = ap.parse_args()
    a.fn(a)
