"""Pitch adjudication of tone disagreements between the label, ToJyutping and the model (held-out clips).

    pip install ToJyutping praat-parselmouth        # or --target <dir> and pass --extra_path <dir>
    python check_tone_pitch.py build   --model_id checkpoints/run21_40000 [--extra_path <dir>]   # decode, track F0, save tmp/pitch_feats.json
    python check_tone_pitch.py analyze_ctx                                                        # context-aware tone classifier, adjudicate (use this)
    python check_tone_pitch.py analyze                                                            # own-contour-only classifier (weaker; for comparison)
    python check_tone_pitch.py wavs --extra_path <dir>                                            # reliability of the pitch classifier on the person-verified wavs
    python check_tone_pitch.py fusion                                                             # model tone posterior x pitch classifier: does fusing help? (no)

build: for every held-out clip (tts validation + test, emo test) the model is decoded greedily; the CTC spikes locate each
predicted syllable, Praat tracks F0 over the syllable's window, and the contour (5 points in semitones relative to the
clip's median F0, plus mean level and duration) is stored together with the label, the ToJyutping reading and the
prediction for that syllable.

analyze: a tone classifier is trained on "control" syllables where label, ToJyutping and model all agree (grouped 5-fold
cross-validation by clip, so it never sees the clip it scores). Its accuracy on the controls, per tone pair, tells how
much pitch can separate two tones; it is then applied to the disputed syllables (tone is the only difference between label
and prediction) to estimate how often the model's tone fits the pitch better than the label's.
"""

import argparse
import json
import os
import sys
import zlib
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

ONSETS = {"b", "c", "d", "f", "g", "gw", "h", "j", "k", "kw", "l", "m", "n", "ng", "p", "s", "t", "w", "z"}
FRAME_S, T0 = 0.04, 0.03          # model output frame: 25 Hz; frame k is centred at about T0 + k * FRAME_S seconds
FEATS_PATH = os.path.join(REPO, "tmp", "pitch_feats.json")


def align_idx(ref, hyp):
    """edit alignment -> list of (op, ref index, hyp index)"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1)); i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", i - 1, None)); i -= 1
        else:
            ops.append(("I", None, j - 1)); j -= 1
    return ops[::-1]


# ----------------------------------------------------------------------------- build
def tone_posteriors(lp, toks, vocab, s):
    """log-posterior of every rime+tone token with the same rime as the predicted one, at the frame of the rime's peak"""
    import numpy as np

    stem = toks[s["kr"]][:-1]
    f = s["r0"] + int(np.argmax(lp[s["r0"]: s["r1"] + 1, s["kr"]]))
    return {t: round(float(lp[f, vocab[stem + t]]), 3) for t in "123456" if stem + t in vocab}


def load_env(args):
    if args.extra_path:
        sys.path.insert(0, args.extra_path)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import librosa
    import numpy as np
    import parselmouth
    import torch
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor

    from ctc_utils import ctc_ids_to_token_text, load_ctc_tokenizer
    from jyutping import nucleus_tone_text_to_jyutping
    from model import Wav2Vec2BertForJyutpingCTC

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    tok = load_ctc_tokenizer(os.path.join(args.model_id, "vocab.json"))
    proc = Wav2Vec2BertProcessor(feature_extractor=SeamlessM4TFeatureExtractor.from_pretrained(args.model_id), tokenizer=tok)
    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        args.model_id, add_adapter=True, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=tok.pad_token_id, vocab_size=len(tok),
    ).to(dev).eval()
    toks = tok.convert_ids_to_tokens(list(range(len(tok))))
    return dict(np=np, torch=torch, parselmouth=parselmouth, librosa=librosa, dev=dev, tok=tok, proc=proc, model=model,
                toks=toks, vocab={t: i for i, t in enumerate(toks)}, decode=ctc_ids_to_token_text, to_jp=nucleus_tone_text_to_jyutping)


def process_clip(env, y, name, ci, gold, conv, skipped):
    """decode one clip, locate its syllables, track F0; returns (clip entry, syllable records) or None"""
    np, torch, model, proc, dev, toks, tok = (env[k] for k in ("np", "torch", "model", "proc", "dev", "toks", "tok"))
    inp = proc(y, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        lg = model(input_features=inp.input_features.to(dev), attention_mask=inp.attention_mask.to(dev), return_dict=True).logits
    n = int(model._get_feat_extract_output_lengths(inp.attention_mask.sum(-1))[0])
    ids = lg[0, :n].argmax(-1).cpu().numpy()
    lp = torch.log_softmax(lg[0, :n].float(), -1).cpu().numpy()
    pred = env["to_jp"](env["decode"](tok, ids.tolist(), n)).split()

    runs, prev = [], -1                     # greedy CTC path -> token runs with frame ranges
    for f, k in enumerate(ids):
        if k != prev:
            if k not in (0, 2):
                runs.append([int(k), f, f])
        elif runs and k not in (0, 2):
            runs[-1][2] = f
        prev = k
    sylls, pend = [], None
    for k, f0, f1 in runs:
        t = toks[k]
        if t in ONSETS:
            pend = (t, f0)
        else:
            sylls.append({"syl": (pend[0] if pend else "") + t, "start": pend[1] if pend else f0, "r0": f0, "r1": f1, "kr": k})
            pend = None
    if [s["syl"] for s in sylls] != pred:
        skipped["syllable assembly differs from the decoder"] += 1
        return None
    snd = env["parselmouth"].Sound(y.astype(np.float64), sampling_frequency=16000)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=450)
    f0, tt = pitch.selected_array["frequency"], pitch.xs()
    voiced = f0 > 0
    if voiced.sum() < 20:
        skipped["too little voiced speech"] += 1
        return None
    ref = float(np.median(f0[voiced]))
    f0st = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-6) / ref), np.nan)
    entry = {
        "t0": float(tt[0]), "f0st": [None if np.isnan(v) else round(float(v), 2) for v in f0st], "dur": len(y) / 16000,
        "sy": [[T0 + FRAME_S * s["start"], T0 + FRAME_S * s["r0"], T0 + FRAME_S * s["r1"]] for s in sylls],
        "tp": [tone_posteriors(lp, toks, env["vocab"], s) for s in sylls],
    }
    recs = [{"set": name, "idx": ci, "i": i, "j": j, "n": len(gold), "L": gold[i], "P": pred[j],
             "C": conv[i] if len(conv) == len(gold) else None}
            for op, i, j in align_idx(gold, pred) if op in ("=", "S")]
    return entry, recs


def build(args):
    env = load_env(args)
    import ToJyutping as tj
    from datasets import Audio

    from check_labels_vs_tojyutping import convert
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from eval_w2v2bert_nucleus_tone import read_audio
    from jyutping import normalize_jyutping_text

    np, librosa = env["np"], env["librosa"]
    sp = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)
    emo = load_any_dataset(args.emo_path)["test"]
    sets = [("tts_val", sp["validation"], "phone"), ("tts_test", sp["test"], "phone"), ("emo_test", emo, "labels")]

    recs, clips, skipped = [], {}, Counter()
    for name, ds, col in sets:
        ds = ds.select_columns(["audio", "text", col]).cast_column("audio", Audio(decode=False))
        for ci, row in enumerate(ds):
            y, sr = read_audio(row["audio"])
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            conv, _ = convert(tj, row["text"])
            out = process_clip(env, y.astype(np.float32), name, ci, normalize_jyutping_text(row[col]).split(), conv, skipped)
            if out is not None:
                clips[f"{name}:{ci}"] = out[0]
                recs += out[1]
        print(f"{name}: {len(ds)} clips done, {len(recs)} syllables so far", flush=True)
    out_path = args.out or FEATS_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"recs": recs, "clips": clips}, open(out_path, "w"))
    print("saved", out_path, "| skipped clips:", dict(skipped), flush=True)


# ----------------------------------------------------------------------------- analyze
def make_feats(clip, j, a, b):
    """5-point semitone contour + mean level + duration of the voiced F0 in [rime spike - a, min(rime spike + b, next syllable)]."""
    import numpy as np
    from scipy.signal import medfilt

    sy = clip["sy"]
    t_r = (sy[j][1] + sy[j][2]) / 2
    t_n = sy[j + 1][0] if j + 1 < len(sy) else 1e9
    lo, hi = t_r - a, min(t_r + b, t_n)
    f = clip["_f"]
    times = clip["t0"] + 0.01 * np.arange(len(f))
    m = ~np.isnan(f) & (times >= lo) & (times < hi)
    if m.sum() < 6:
        return None
    x, st = times[m], medfilt(f[m], 3)
    pts = np.interp(np.linspace(x[0], x[-1], 5), x, st)
    return [float(v) for v in pts] + [float(st.mean()), float(x[-1] - x[0])]


_T3_LEVELS = {}


def t3_levels(clips, pmap, name, ci, a, b):
    """mean pitch level (semitones re the clip median) of every syllable the model predicted as tone 3, by syllable index"""
    key = (name, ci, a, b)
    if key not in _T3_LEVELS:
        clip = clips[f"{name}:{ci}"]
        lv = {}
        for j in range(len(clip["sy"])):
            if pmap.get((name, ci, j), "0")[-1] == "3":
                f = make_feats(clip, j, a, b)
                if f is not None:
                    lv[j] = f[5]
        _T3_LEVELS[key] = lv
    return _T3_LEVELS[key]


def ctx_vector(clips, pmap, r, a, b, context=True, anchor=False):
    """own contour + position (+ neighbours' pitch, predicted tone and spacing) for one syllable record.
    anchor=True expresses every pitch relative to the clip's other tone-3 syllables (the listener's reference level),
    leaving out the syllable itself so that it cannot anchor on itself."""
    import numpy as np

    nan = float("nan")
    clip = clips[f"{r['set']}:{r['idx']}"]
    own = make_feats(clip, r["j"], a, b)
    if own is None:
        return None
    shift, has = 0.0, 0.0
    if anchor:
        others = [v for jj, v in t3_levels(clips, pmap, r["set"], r["idx"], a, b).items() if jj != r["j"]]
        if others:
            shift, has = float(np.median(others)), 1.0
        own = [x - shift for x in own[:6]] + [own[6]]
    v = own + [r["i"] / max(r["n"], 1), float(r["i"] == 0), float(r["i"] == r["n"] - 1)] + ([has] if anchor else [])
    if not context:
        return v
    sy, j = clip["sy"], r["j"]
    for jj in (j - 1, j + 1):
        f = make_feats(clip, jj, a, b) if 0 <= jj < len(sy) else None
        v += [f[5] - shift, f[0] - shift, f[4] - shift] if f else [nan, nan, nan]
        key = (r["set"], r["idx"], jj)
        v.append(float(pmap[key][-1]) if key in pmap else 0.0)
        v.append(abs(sy[jj][1] - sy[j][1]) if 0 <= jj < len(sy) else nan)
    return v


def analyze(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from error_consistency import parse

    blob = json.load(open(FEATS_PATH))
    recs, clips = blob["recs"], blob["clips"]
    for c in clips.values():
        c["_f"] = np.array([np.nan if v is None else v for v in c["f0st"]], dtype=float)
    rng = np.random.default_rng(0)
    fold = lambda r: zlib.crc32(f"{r['set']}:{r['idx']}".encode()) % 5
    tone = lambda s: s[-1]

    ctrl_all = [r for r in recs if r["C"] and r["L"] == r["P"] == r["C"]]
    cand_all = [r for r in recs if r["L"] != r["P"]]

    def featurize(rows, a, b):
        out = []
        for r in rows:
            f = make_feats(clips[f"{r['set']}:{r['idx']}"], r["j"], a, b)
            if f is not None:
                out.append(dict(r, feats=f))
        return out

    def run_cv(ctrl, cand):
        proba = {}
        for k in range(5):
            tr = [r for r in ctrl if fold(r) != k]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit([r["feats"] + [r["i"] / max(r["n"], 1)] for r in tr], [tone(r["L"]) for r in tr])
            te = [r for r in ctrl if fold(r) == k] + [r for r in cand if fold(r) == k]
            if te:
                for r, p in zip(te, clf.predict_proba([r["feats"] + [r["i"] / max(r["n"], 1)] for r in te])):
                    proba[id(r)] = dict(zip(clf.classes_, p))
        acc = np.mean([max(proba[id(r)], key=proba[id(r)].get) == tone(r["L"]) for r in ctrl])
        return acc, proba

    # 1. calibrate the window on the controls (spike time -> where the syllable's pitch really is)
    sample = [r for r in ctrl_all if zlib.crc32(f"{r['set']}:{r['idx']}".encode()) % 2 == 0]   # half of the clips is enough
    print("1. Calibrating the pitch window relative to the CTC rime spike (6-way tone accuracy on control syllables, 5-fold by clip)")
    print(f"   {'start before spike':>20} " + " ".join(f"end +{b:.2f}s".rjust(10) for b in (0.0, 0.08, 0.16, 0.24)))
    best = (0, None)
    for a in (0.0, 0.08, 0.16, 0.24, 0.32, 0.40):
        row = []
        for b in (0.0, 0.08, 0.16, 0.24):
            ctrl = featurize(sample, a, b)
            if len(ctrl) < 500:
                row.append("   n/a"); continue
            acc, _ = run_cv(ctrl, [])
            row.append(f"{100 * acc:5.1f}%/{len(ctrl) // 1000}k")
            if acc > best[0]:
                best = (acc, (a, b))
        print(f"   {a:>19.2f}s " + " ".join(x.rjust(10) for x in row))
    a_best, b_best = best[1]
    print(f"   best window: from {a_best:.2f} s before to {b_best:.2f} s after the rime spike (or the next syllable) -> {100 * best[0]:.1f}%")

    ctrl, cand = featurize(ctrl_all, a_best, b_best), featurize(cand_all, a_best, b_best)
    acc, proba = run_cv(ctrl, cand)
    base = Counter(tone(r["L"]) for r in ctrl).most_common(1)[0][1] / len(ctrl)
    print(f"\n   usable syllables: {len(ctrl)} controls of {len(ctrl_all)}, {len(cand)} disputed of {len(cand_all)}")
    print(f"   6-way accuracy with this window {100 * acc:.1f}% (always guessing the commonest tone: {100 * base:.1f}%)")
    by = defaultdict(list)
    for r in ctrl:
        by[tone(r["L"])].append(r["feats"])
    print("   mean contour per tone (semitones re clip median, 5 points) and the share classified correctly:")
    for t in sorted(by):
        rs = [r for r in ctrl if tone(r["L"]) == t]
        ok = np.mean([max(proba[id(r)], key=proba[id(r)].get) == t for r in rs])
        print(f"     T{t} n={len(rs):5} " + " ".join(f"{v:+5.1f}" for v in np.array(by[t])[:, :5].mean(0)) + f"   correct {100 * ok:3.0f}%")

    def pair_acc(a, b):
        out = []
        for t, o in ((a, b), (b, a)):
            rs = [r for r in ctrl if tone(r["L"]) == t]
            if rs:
                out.append(np.mean([proba[id(r)][t] > proba[id(r)][o] for r in rs]))
        return float(np.mean(out)) if out else None

    print("\n   pairwise accuracy (how well pitch separates two tones):  " + "  ".join(
        f"T{a}/T{b} {100 * pair_acc(a, b):.0f}%" for a, b in (("3", "6"), ("4", "6"), ("2", "5"), ("1", "3"), ("2", "3"), ("5", "6"))))

    pops = {
        "A  label deviates from ToJyutping, model sides with ToJyutping": lambda r: r["C"] and r["L"] != r["C"] and r["P"] == r["C"],
        "B  label = ToJyutping, model deviates": lambda r: r["C"] and r["L"] == r["C"] and r["P"] != r["L"],
        "C  label != ToJyutping, model differs from both": lambda r: r["C"] and r["L"] != r["C"] and r["P"] != r["C"],
    }
    print("\n2. Disputed syllables: does the pitch fit the model's tone or the label's tone better?")
    print("   (tone is the only difference; q = estimated share of cases where the model's tone is the acoustically correct one,")
    print("   corrected for the classifier's accuracy on the controls; it is only meaningful when that accuracy is well above 50%)")
    rows, pooled = [], []
    for label, sel in pops.items():
        allc = [r for r in cand if sel(r)]
        tone_only = [r for r in allc if parse(r["L"])[:3] == parse(r["P"])[:3] and tone(r["L"]) != tone(r["P"])]
        pairs = []
        for r in tone_only:
            pL, pP = proba[id(r)][tone(r["L"])], proba[id(r)][tone(r["P"])]
            rows.append((label[0], r["set"], r["idx"], r["i"], r["L"], r["C"], r["P"], round(pL, 3), round(pP, 3)))
            pairs.append((tone(r["L"]), tone(r["P"]), pP > pL, pair_acc(tone(r["L"]), tone(r["P"]))))
        if label[0] in "AB":
            pooled += pairs
        n = len(pairs)
        print(f"\n   [{label}]  disputed syllables with pitch: {len(allc)}; tone-only: {n}")
        if n < 8:
            continue
        fav = np.array([p[2] for p in pairs], dtype=float)
        acc_p = np.array([p[3] if p[3] is not None else 0.5 for p in pairs])
        est = lambda f, a: min(max((f.mean() - (1 - a.mean())) / (2 * a.mean() - 1), 0.0), 1.0) if a.mean() > 0.55 else float("nan")
        boots = [est(fav[ix], acc_p[ix]) for ix in (rng.integers(0, n, n) for _ in range(2000))]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if not np.all(np.isnan(boots)) else (float("nan"),) * 2
        print(f"      pitch favours the model's tone in {100 * fav.mean():.0f}% of the cases (50% = no information); "
              f"classifier accuracy on controls for these tone pairs {100 * acc_p.mean():.0f}%")
        print(f"      estimated share where the model is right: q = {100 * est(fav, acc_p):.0f}% (95% interval {100 * lo:.0f}% to {100 * hi:.0f}%)")
        bypair = defaultdict(list)
        for a_, b_, f_, acc_ in pairs:
            bypair[(a_, b_)].append((f_, acc_))
        top = sorted(bypair.items(), key=lambda kv: -len(kv[1]))[:6]
        print("      by pair (label tone -> model tone: n, pitch favours model, control accuracy): " + "; ".join(
            f"T{a_}->T{b_}: {len(v)}, {100 * np.mean([x[0] for x in v]):.0f}%, {100 * (v[0][1] or 0.5):.0f}%" for (a_, b_), v in top))
    if pooled:
        n = len(pooled)
        fav = np.array([p[2] for p in pooled], dtype=float)
        acc_p = np.array([p[3] if p[3] is not None else 0.5 for p in pooled])
        est = lambda f, a: min(max((f.mean() - (1 - a.mean())) / (2 * a.mean() - 1), 0.0), 1.0)
        lo, hi = np.percentile([est(fav[ix], acc_p[ix]) for ix in (rng.integers(0, n, n) for _ in range(2000))], [2.5, 97.5])
        print(f"\n   [A + B pooled: every tone-only disagreement between label and model that has a pitch contour]  n = {n}")
        print(f"      pitch favours the model's tone in {100 * fav.mean():.0f}% of the cases; classifier accuracy on controls {100 * acc_p.mean():.0f}%; "
              f"q = {100 * est(fav, acc_p):.0f}% (95% interval {100 * lo:.0f}% to {100 * hi:.0f}%)")
    out = os.path.join(REPO, "tmp", "pitch_cases.csv")
    with open(out, "w") as f:
        f.write("population,set,clip,syllable_index,label,toJyutping,model,p_label_tone,p_model_tone\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"\nper-case results written to {out}")


def analyze_context(args):
    """Same adjudication, but the classifier also sees the neighbouring syllables (pitch, predicted tone, timing, position)
    and the calibration is direction-specific: for a disputed pair (label tone a, model tone b) the classifier's chance of
    preferring a when a is true (sa) and of preferring b when b is true (sb) are measured separately on the controls, so a
    classifier that leans towards one tone cannot make that tone look right. q (share of cases where the model's tone is the
    acoustically right one) is the maximum-likelihood value given those sensitivities."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from error_consistency import parse

    a, b = args.a, args.b
    blob = json.load(open(FEATS_PATH))
    recs, clips = blob["recs"], blob["clips"]
    for c in clips.values():
        c["_f"] = np.array([np.nan if v is None else v for v in c["f0st"]], dtype=float)
    pmap = {(r["set"], r["idx"], r["j"]): r["P"] for r in recs}
    rng = np.random.default_rng(0)
    tone = lambda s: s[-1]
    fold = lambda r: zlib.crc32(f"{r['set']}:{r['idx']}".encode()) % 5
    nan = float("nan")

    def vec(r, context):
        return ctx_vector(clips, pmap, r, a, b, context, anchor=args.anchor == "t3")

    ctrl_all = [r for r in recs if r["C"] and r["L"] == r["P"] == r["C"]]
    cand_all = [r for r in recs if r["L"] != r["P"]]

    def run(context):
        ctrl = [dict(r, x=x) for r in ctrl_all if (x := vec(r, context)) is not None]
        cand = [dict(r, x=x) for r in cand_all if (x := vec(r, context)) is not None]
        proba = {}
        for k in range(5):
            tr = [r for r in ctrl if fold(r) != k]
            if context:
                clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, random_state=0)
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(np.array([r["x"] for r in tr], dtype=float), [tone(r["L"]) for r in tr])
            te = [r for r in ctrl if fold(r) == k] + [r for r in cand if fold(r) == k]
            if te:
                for r, pr in zip(te, clf.predict_proba(np.array([r["x"] for r in te], dtype=float))):
                    proba[id(r)] = dict(zip(clf.classes_, pr))
        acc = np.mean([max(proba[id(r)], key=proba[id(r)].get) == tone(r["L"]) for r in ctrl])
        return ctrl, cand, proba, acc

    results = {}
    for name, context in (("own contour only", False), ("with context", True)):
        ctrl, cand, proba, acc = run(context)
        results[name] = (ctrl, cand, proba, acc)
        print(f"{name:18} 6-way tone accuracy on {len(ctrl)} control syllables: {100 * acc:.1f}%")
    ctrl, cand, proba, acc = results["with context"]

    def sens(t, o):
        rs = [r for r in ctrl if tone(r["L"]) == t]
        return float(np.mean([proba[id(r)][t] > proba[id(r)][o] for r in rs])) if rs else 0.5

    print("\nDirection-specific sensitivities on the controls (chance of preferring tone X over tone Y when X is true):")
    print("   " + "  ".join(f"T{x}>T{y} {100 * sens(x, y):.0f}%" for x, y in
                            (("3", "6"), ("6", "3"), ("4", "6"), ("6", "4"), ("2", "5"), ("5", "2"), ("1", "3"), ("3", "1"))))

    def mle(favs, sa, sb):
        qs = np.linspace(0, 1, 1001)
        pf = np.clip(qs[:, None] * sb[None, :] + (1 - qs[:, None]) * (1 - sa[None, :]), 1e-6, 1 - 1e-6)
        ll = (favs[None, :] * np.log(pf) + (1 - favs[None, :]) * np.log(1 - pf)).sum(1)
        return qs[int(np.argmax(ll))]

    pops = {
        "A  label deviates from ToJyutping, model sides with ToJyutping": lambda r: r["C"] and r["L"] != r["C"] and r["P"] == r["C"],
        "B  label = ToJyutping, model deviates": lambda r: r["C"] and r["L"] == r["C"] and r["P"] != r["L"],
    }
    print("\nDisputed syllables, tone the only difference; q = share of cases where the model's tone is the acoustically right one")
    allcases = []
    for label, sel in pops.items():
        cases = []
        for r in cand:
            if sel(r) and parse(r["L"])[:3] == parse(r["P"])[:3] and tone(r["L"]) != tone(r["P"]):
                tl, tp = tone(r["L"]), tone(r["P"])
                cases.append((proba[id(r)][tp] > proba[id(r)][tl], sens(tl, tp), sens(tp, tl), tl, tp))
        allcases += cases
        fav = np.array([c[0] for c in cases], dtype=float); sa = np.array([c[1] for c in cases]); sb = np.array([c[2] for c in cases])
        n = len(cases)
        boots = [mle(fav[ix], sa[ix], sb[ix]) for ix in (rng.integers(0, n, n) for _ in range(300))]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"   [{label}] n = {n}: pitch favours the model in {100 * fav.mean():.0f}%; q = {100 * mle(fav, sa, sb):.0f}% (95% interval {100 * lo:.0f}% to {100 * hi:.0f}%)")
    fav = np.array([c[0] for c in allcases], dtype=float); sa = np.array([c[1] for c in allcases]); sb = np.array([c[2] for c in allcases])
    n = len(allcases)
    boots = [mle(fav[ix], sa[ix], sb[ix]) for ix in (rng.integers(0, n, n) for _ in range(300))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"   [A + B pooled] n = {n}: pitch favours the model in {100 * fav.mean():.0f}%; q = {100 * mle(fav, sa, sb):.0f}% (95% interval {100 * lo:.0f}% to {100 * hi:.0f}%)")
    by = defaultdict(list)
    for f_, x, y, tl, tp in allcases:
        by[(tl, tp)].append((f_, x, y))
    print("   by pair (label tone -> model tone: n, pitch favours model, sensitivities sa/sb): " + "; ".join(
        f"T{k[0]}->T{k[1]}: {len(v)}, {100 * np.mean([z[0] for z in v]):.0f}%, {100 * v[0][1]:.0f}/{100 * v[0][2]:.0f}"
        for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:6]))


def fusion(args):
    """Late fusion: model tone posterior (same rime, every tone) x pitch-classifier tone probability. Does the fused tone agree
    with the label more often than the model's own tone? Gamma is tuned on tts_val and applied unchanged to tts_test / emo_test."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    a, b = args.a, args.b
    blob = json.load(open(FEATS_PATH))
    recs, clips = blob["recs"], blob["clips"]
    for c in clips.values():
        c["_f"] = np.array([np.nan if v is None else v for v in c["f0st"]], dtype=float)
    pmap = {(r["set"], r["idx"], r["j"]): r["P"] for r in recs}
    tone = lambda s: s[-1]
    fold = lambda r: zlib.crc32(f"{r['set']}:{r['idx']}".encode()) % 5
    allr = [dict(r, x=x) for r in recs if (x := ctx_vector(clips, pmap, r, a, b)) is not None]
    ctrl = [r for r in allr if r["C"] and r["L"] == r["P"] == r["C"]]
    for k in range(5):                                          # out-of-fold pitch probabilities for every syllable
        tr = [r for r in ctrl if fold(r) != k]
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, random_state=0)
        clf.fit(np.array([r["x"] for r in tr], dtype=float), [tone(r["L"]) for r in tr])
        te = [r for r in allr if fold(r) == k]
        for r, pr in zip(te, clf.predict_proba(np.array([r["x"] for r in te], dtype=float))):
            r["pp"] = dict(zip(clf.classes_, pr))
    rows = []
    for r in allr:
        tp = clips[f"{r['set']}:{r['idx']}"]["tp"][r["j"]]
        if len(tp) >= 2 and tone(r["P"]) in tp:
            rows.append(dict(r, tp=tp))
    print(f"syllables with a pitch contour and at least two tone candidates for the predicted rime: {len(rows)} of {len(recs)}")

    def fused(r, g):
        return max(r["tp"], key=lambda t: r["tp"][t] + g * np.log(max(r["pp"].get(t, 1e-4), 1e-4)))

    def pop(r):
        if not r["C"]:
            return "no ToJyutping"
        if r["L"] == r["C"]:
            return "label = ToJyutping"
        return "label deviates"

    sets = ("tts_val", "tts_test", "emo_test")
    print("\nErrors against the label among these syllables (model alone -> fused), by gamma (weight of the pitch classifier)")
    print(f"{'gamma':>6} " + " ".join(f"{s:>22}" for s in sets))
    base = {s: sum(r["P"] != r["L"] for r in rows if r["set"] == s) for s in sets}
    table = {}
    for g in (0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        cells = []
        for s_ in sets:
            e = sum((r["P"][:-1] + fused(r, g)) != r["L"] for r in rows if r["set"] == s_)
            table[(g, s_)] = e
            cells.append(f"{base[s_]:>5} -> {e:>5} ({e - base[s_]:+d})".rjust(22))
        print(f"{g:>6} " + " ".join(cells))
    best = min((0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0), key=lambda g: table[(g, "tts_val")])
    print(f"\ngamma tuned on tts_val: {best}")
    for s_ in sets[1:]:
        sub = [r for r in rows if r["set"] == s_]
        fixed = sum(r["P"] != r["L"] and (r["P"][:-1] + fused(r, best)) == r["L"] for r in sub)
        broke = sum(r["P"] == r["L"] and (r["P"][:-1] + fused(r, best)) != r["L"] for r in sub)
        changed = sum(fused(r, best) != tone(r["P"]) for r in sub)
        print(f"   {s_:9} tone changed in {changed} of {len(sub)} syllables: {fixed} errors fixed, {broke} correct syllables broken, net {broke - fixed:+d}")
        byp = defaultdict(lambda: [0, 0])
        for r in sub:
            new = r["P"][:-1] + fused(r, best)
            if r["P"] != new:
                byp[pop(r)][0] += (r["P"] != r["L"] and new == r["L"])
                byp[pop(r)][1] += (r["P"] == r["L"] and new != r["L"])
        print("              by label type (fixed / broken): " + "; ".join(f"{k}: {v[0]} / {v[1]}" for k, v in byp.items()))
    tone_only_err = {s_: sum(1 for r in rows if r["set"] == s_ and r["P"] != r["L"] and r["P"][:-1] == r["L"][:-1] and tone(r["L"]) in r["tp"]) for s_ in sets}
    print(f"\nupper bound: tone-only errors where the label's tone is among the candidates (a perfect tone chooser fixes all of them): {tone_only_err}")


def wavs_check(args):
    """Is the pitch classifier reliable on real recordings whose tones were checked by a person? Applies the classifier
    (trained on the held-out control syllables) to the hand-labelled wavs in tmp/labels.md."""
    import re

    env = load_env(args)
    np = env["np"]
    from sklearn.ensemble import HistGradientBoostingClassifier

    from jyutping import normalize_jyutping_text

    a, b = args.a, args.b
    tone = lambda s: s[-1]
    fold = lambda r: zlib.crc32(f"{r['set']}:{r['idx']}".encode()) % 5
    blob = json.load(open(FEATS_PATH))
    recs, clips = blob["recs"], blob["clips"]
    for c in clips.values():
        c["_f"] = np.array([np.nan if v is None else v for v in c["f0st"]], dtype=float)
    pmap = {(r["set"], r["idx"], r["j"]): r["P"] for r in recs}
    anc = args.anchor == "t3"
    ctrl = [dict(r, x=x) for r in recs if r["C"] and r["L"] == r["P"] == r["C"] and (x := ctx_vector(clips, pmap, r, a, b, True, anc)) is not None]
    mk = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, random_state=0)
    X = lambda rows: np.array([r["x"] for r in rows], dtype=float)
    clf = mk().fit(X(ctrl), [tone(r["L"]) for r in ctrl])
    cv = {}                                                         # in-domain out-of-fold probabilities, for the reliability curve
    for k in range(5):
        tr = [r for r in ctrl if fold(r) != k]
        c5 = mk().fit(X(tr), [tone(r["L"]) for r in tr])
        te = [r for r in ctrl if fold(r) == k]
        for r, pr in zip(te, c5.predict_proba(X(te))):
            cv[id(r)] = dict(zip(c5.classes_, pr))

    wav_items = []
    for line in open(os.path.join(REPO, "tmp", "labels.md"), encoding="utf-8"):
        m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|", line)
        if m and m.group(1).endswith(".wav"):
            y = env["librosa"].load(os.path.join(REPO, "tmp", m.group(1)), sr=16000, mono=True)[0].astype(np.float32)
            wav_items.append((m.group(1), y, normalize_jyutping_text(m.group(2)).split()))
    clips_w, recs_w, skipped = {}, [], Counter()
    for k, (fname, y, gold) in enumerate(wav_items):
        out = process_clip(env, y, "wavs", k, gold, [], skipped)
        if out is not None:
            clips_w[f"wavs:{k}"] = out[0]
            recs_w += out[1]
    for c in clips_w.values():
        c["_f"] = np.array([np.nan if v is None else v for v in c["f0st"]], dtype=float)
    pmap_w = {(r["set"], r["idx"], r["j"]): r["P"] for r in recs_w}
    rows = []
    for r in recs_w:
        x = ctx_vector(clips_w, pmap_w, r, a, b, True, anc)
        if x is not None:
            rows.append(dict(r, x=x, pp=dict(zip(clf.classes_, clf.predict_proba(np.array([x], dtype=float))[0]))))
    print(f"wavs: {len(wav_items)} clips, {len(recs_w)} aligned syllables, {len(rows)} with a usable pitch contour (skipped clips: {dict(skipped)})\n")

    def wilson(k, n):
        if n == 0:
            return (0.0, 0.0)
        z, p = 1.96, k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
        return c - h, c + h

    top = lambda pp: max(pp, key=pp.get)
    ok_c = sum(top(r["pp"]) == tone(r["L"]) for r in rows)
    ok_m = sum(tone(r["P"]) == tone(r["L"]) for r in rows)
    lo, hi = wilson(ok_c, len(rows))
    print("1. Tone accuracy against the person-verified labels")
    print(f"   pitch classifier   {ok_c}/{len(rows)} = {100 * ok_c / len(rows):.0f}% (95% interval {100 * lo:.0f}-{100 * hi:.0f}%)   [in-domain controls: 69.6%]")
    print(f"   ASR model's tone   {ok_m}/{len(rows)} = {100 * ok_m / len(rows):.0f}%")
    print("\n2. How reliable is the pitch classifier when it is confident? (accuracy against the labels / share of syllables)")
    print(f"   {'confidence':>12} {'in-domain controls (pseudo-labels)':>38} {'wavs (verified labels)':>26}")
    for th in (0.0, 0.5, 0.7, 0.9):
        cs = [r for r in ctrl if max(cv[id(r)].values()) >= th]
        ws = [r for r in rows if max(r["pp"].values()) >= th]
        cacc = sum(top(cv[id(r)]) == tone(r["L"]) for r in cs) / len(cs)
        wacc = f"{sum(top(r['pp']) == tone(r['L']) for r in ws)}/{len(ws)} = {100 * sum(top(r['pp']) == tone(r['L']) for r in ws) / len(ws):.0f}%" if ws else "-"
        print(f"   {'>= ' + str(th):>12} {f'{100 * cacc:.0f}% of {len(cs)} ({100 * len(cs) / len(ctrl):.0f}% of syllables)':>38} {wacc:>20} ({len(ws)}/{len(rows)} syllables)")
    print("\n3. Where the ASR model's tone differs from the verified tone (pitch should side with the verified one)")
    bad = [r for r in rows if tone(r["P"]) != tone(r["L"])]
    for r in bad:
        pl, pm = r["pp"][tone(r["L"])], r["pp"].get(tone(r["P"]), 0.0)
        print(f"   wav {r['idx']} syllable {r['i']}: verified {r['L']}, model {r['P']}; pitch classifier says T{top(r['pp'])} (p {max(r['pp'].values()):.2f}); "
              f"p(verified tone) {pl:.2f} vs p(model tone) {pm:.2f} -> pitch sides with the {'VERIFIED label' if pl > pm else 'MODEL'}")
    if bad:
        print(f"   pitch sides with the verified label in {sum(r['pp'][tone(r['L'])] > r['pp'].get(tone(r['P']), 0) for r in bad)} of {len(bad)} model tone errors")
    good = [r for r in rows if tone(r["P"]) == tone(r["L"])]
    wrong = [r for r in good if top(r["pp"]) != tone(r["L"])]
    print(f"   where the model's tone is right ({len(good)} syllables), the pitch classifier would have overruled it with a wrong tone in {len(wrong)} ({100 * len(wrong) / max(len(good), 1):.0f}%)")
    print("\n4. By verified tone (pitch classifier correct / n): " + "  ".join(
        f"T{t} {sum(top(r['pp']) == t for r in rows if tone(r['L']) == t)}/{sum(tone(r['L']) == t for r in rows)}" for t in "123456"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    b.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    b.add_argument("--extra_path", default=None, help="directory holding ToJyutping and praat-parselmouth if installed with --target")
    b.add_argument("--out", default=None, help="where to write the features (default tmp/pitch_feats.json)")
    sub.add_parser("analyze")
    w = sub.add_parser("wavs")
    w.add_argument("--model_id", default=os.path.join(REPO, "checkpoints", "run21_40000"))
    w.add_argument("--extra_path", default=None)
    w.add_argument("--a", type=float, default=0.16)
    w.add_argument("--b", type=float, default=0.0)
    w.add_argument("--anchor", choices=["clip_median", "t3"], default="clip_median")
    f = sub.add_parser("fusion")
    f.add_argument("--a", type=float, default=0.16)
    f.add_argument("--b", type=float, default=0.0)
    c = sub.add_parser("analyze_ctx")
    c.add_argument("--a", type=float, default=0.16, help="window starts this many seconds before the rime spike")
    c.add_argument("--b", type=float, default=0.0, help="window ends this many seconds after the rime spike")
    c.add_argument("--anchor", choices=["clip_median", "t3"], default="clip_median", help="pitch reference: the clip median or its other tone-3 syllables")
    args = ap.parse_args()
    {"build": build, "analyze": analyze, "analyze_ctx": analyze_context, "fusion": fusion, "wavs": wavs_check}[args.cmd](args)


if __name__ == "__main__":
    main()
