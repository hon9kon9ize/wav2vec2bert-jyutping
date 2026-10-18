"""Are the onset / rime / tone errors consistent across datasets? (dataset-specific overfitting check)

Step 1 - decode a model on several sets (batch size 1, length-aware CTC decode); writes tmp/preds_<name>.json:

    python error_consistency.py decode --kind ours --model_id checkpoints/run21_40000 --name run21
    python error_consistency.py decode --kind ours --model_id checkpoints/run20_best --name run20
    python error_consistency.py decode --kind published --model_id <published_dir> --name published

Sets: tts_val, tts_test (the fixed split), tts_train, emo_train (random --n_train clips of the training split) and emo_test.

Step 2 - compare; every --model gives the dataset families the model was trained on:

    python error_consistency.py report --model run21=tts+emo --model run20=tts --model published=emo

The report shows, per model and dataset, the share of syllables whose onset / rime / tone is wrong (95% clip-level
bootstrap intervals), the train-vs-held-out gap per part, whether the two datasets agree on which parts and which
tokens fail, and how each part degrades on a dataset the model never saw.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

TONE_MERGERS = {frozenset(p) for p in (("3", "6"), ("4", "6"), ("2", "5"))}
ONSET_MERGERS = {frozenset(p) for p in (("n", "l"), ("ng", "-"), ("gw", "g"))}
PARTS = ("onset", "nuc", "coda", "tone")
ALL_SETS = ("tts_val", "tts_test", "tts_train", "emo_train", "emo_test")


# ----------------------------------------------------------------------------- syllables
def parse(syl):
    """toned syllable -> (onset, nucleus, coda, tone); a syllabic nasal is a nucleus without onset."""
    if not syl or syl[-1] not in "123456":
        return "", syl, "", "0"
    tone, body = syl[-1], syl[:-1]
    if body in ("m", "ng"):
        return "", body, "", tone
    m = re.match(r"^(ng|gw|kw|[bpmfdtnlgkhwzcsj])?(.*)$", body)
    onset, rime = m.group(1) or "", m.group(2)
    if rime.startswith("yu"):
        return onset, "yu", rime[2:], tone
    coda = ""
    for c in ("ng", "p", "t", "k", "m", "n", "i", "u"):
        if len(rime) > len(c) and rime.endswith(c):
            coda, rime = c, rime[: -len(c)]
            break
    return onset, rime, coda, tone


def align(ref, hyp):
    """Syllable-level edit alignment -> list of (op, ref_index, ref_syl, hyp_syl); op in = S D I."""
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
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", i - 1, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", i - 1, ref[i - 1], None)); i -= 1
        else:
            ops.append(("I", None, None, hyp[j - 1])); j -= 1
    return ops[::-1]


# ----------------------------------------------------------------------------- step 1: decode
def load_sets(sets, emo_path, n_train, seed):
    from datasets import Audio

    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split

    out = {}
    if any(s.startswith("tts") for s in sets):
        sp = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)
        for name, key in (("tts_val", "validation"), ("tts_test", "test"), ("tts_train", "train")):
            if name in sets:
                ds = sp[key]
                if name == "tts_train":
                    ds = ds.shuffle(seed=seed).select(range(n_train))
                out[name] = (ds.select_columns(["audio", "phone"]).cast_column("audio", Audio(decode=False)), "phone")
    if any(s.startswith("emo") for s in sets):
        emo = load_any_dataset(emo_path)
        for name, key in (("emo_test", "test"), ("emo_train", "train")):
            if name in sets:
                ds = emo[key]
                if name == "emo_train":
                    ds = ds.shuffle(seed=seed).select(range(n_train))
                out[name] = (ds.select_columns(["audio", "labels"]).cast_column("audio", Audio(decode=False)), "labels")
    return out


def make_predictor(kind, model_id):
    import torch
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor

    from ctc_utils import ctc_ids_to_token_text, load_ctc_tokenizer
    from jyutping import nucleus_tone_text_to_jyutping

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "ours":
        from model import Wav2Vec2BertForJyutpingCTC

        tok = load_ctc_tokenizer(os.path.join(model_id, "vocab.json"))
        proc = Wav2Vec2BertProcessor(feature_extractor=SeamlessM4TFeatureExtractor.from_pretrained(model_id), tokenizer=tok)
        model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
            model_id, add_adapter=True, ctc_loss_reduction="mean", ctc_zero_infinity=True,
            pad_token_id=tok.pad_token_id, vocab_size=len(tok),
        ).to(dev).eval()

        def predict(y):
            inp = proc(y, sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                lg = model(input_features=inp.input_features.to(dev), attention_mask=inp.attention_mask.to(dev), return_dict=True).logits
            n = int(model._get_feat_extract_output_lengths(inp.attention_mask.sum(-1))[0])
            return nucleus_tone_text_to_jyutping(ctc_ids_to_token_text(tok, lg[0].argmax(-1).tolist(), n)).split()

        return predict

    from model import Wav2Vec2BertForCantonese

    tok = load_ctc_tokenizer(os.path.join(REPO, "vocab.json"))
    tone_tok = load_ctc_tokenizer(os.path.join(REPO, "tone_vocab.json"))
    proc = Wav2Vec2BertProcessor(feature_extractor=SeamlessM4TFeatureExtractor.from_pretrained(model_id), tokenizer=tok)
    model = Wav2Vec2BertForCantonese.from_pretrained(
        model_id, attention_dropout=0.2, hidden_dropout=0.2, feat_proj_dropout=0.0, mask_time_prob=0.0, layerdrop=0.0,
        add_adapter=True, ctc_loss_reduction="mean", pad_token_id=tok.pad_token_id, vocab_size=len(tok),
    ).to(dev).eval()

    def predict_published(y):
        inp = proc(y, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            out, _, _ = model.inference(processor=proc, tone_tokenizer=tone_tok,
                                        input_features=inp.input_features.to(dev), attention_mask=inp.attention_mask.to(dev))
        return out.split()

    return predict_published


def cmd_decode(args):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import librosa
    import numpy as np
    import torch

    from eval_w2v2bert_nucleus_tone import read_audio
    from jyutping import normalize_jyutping_text

    torch.set_num_threads(args.threads)
    sets = [s for s in args.sets.split(",")]
    data = load_sets(sets, args.emo_path, args.n_train, args.seed)
    predict = make_predictor(args.kind, args.model_id)
    out = []
    for name in sets:
        ds, col = data[name]
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        for i, row in enumerate(ds):
            y, sr = read_audio(row["audio"])
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            y = y.astype(np.float32)
            out.append({"set": name, "idx": i, "dur": len(y) / 16000,
                        "gold": normalize_jyutping_text(row[col]).split(), "pred": predict(y)})
        print(f"{args.name}: {name} done ({len(ds)} clips)", flush=True)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    path = f"{args.out_prefix}{args.name}.json"
    json.dump(out, open(path, "w"), ensure_ascii=False)
    print("saved", path, flush=True)


# ----------------------------------------------------------------------------- step 2: statistics
KEYS = ("n", "err", "sub", "onset", "rime", "tone", "merger", "o1", "r1", "t1", "multi")
KI = {k: i for i, k in enumerate(KEYS)}


def clip_stats(c):
    s = dict.fromkeys(KEYS, 0)
    s["n"] = len(c["gold"])
    for op, _, r, h in align(c["gold"], c["pred"]):
        if op == "=":
            continue
        s["err"] += 1
        if op != "S":
            continue
        s["sub"] += 1
        po, pn, pc, pt = parse(r)
        ho, hn, hc, ht = parse(h)
        o, ri, t = po != ho, (pn != hn or pc != hc), pt != ht
        s["onset"] += o; s["rime"] += ri; s["tone"] += t
        k = int(o) + int(ri) + int(t)
        if k == 1:
            s["o1" if o else "r1" if ri else "t1"] += 1
        else:
            s["multi"] += 1
        only_merger = (not ri) and (not o or frozenset((po or "-", ho or "-")) in ONSET_MERGERS) \
            and (not t or frozenset((pt, ht)) in TONE_MERGERS)
        s["merger"] += only_merger
    return s


def token_tables(clips):
    n = {k: Counter() for k in PARTS}
    e = {k: Counter() for k in PARTS}
    conf = {k: Counter() for k in PARTS}
    for c in clips:
        for op, _, r, h in align(c["gold"], c["pred"]):
            if op not in ("=", "S"):
                continue
            po, pn, pc, pt = parse(r)
            gold = {"onset": po or "-", "nuc": pn, "coda": pc or "-", "tone": pt}
            if op == "S":
                ho, hn, hc, ht = parse(h)
                pred = {"onset": ho or "-", "nuc": hn, "coda": hc or "-", "tone": ht}
            for k in PARTS:
                n[k][gold[k]] += 1
                if op == "S" and gold[k] != pred[k]:
                    e[k][gold[k]] += 1
                    conf[k][(gold[k], pred[k])] += 1
    return n, e, conf


def arr_of(clips):
    import numpy as np

    return np.array([[clip_stats(c)[k] for k in KEYS] for c in clips], dtype=np.int64)


def boot(arr, num, den, B, rng):
    import numpy as np

    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    sums = arr[idx].sum(axis=1)
    return sums[:, KI[num]] / np.maximum(sums[:, KI[den]], 1)


def fmt(arr, num, den="n", B=1000, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    p = arr[:, KI[num]].sum() / max(arr[:, KI[den]].sum(), 1)
    lo, hi = np.percentile(boot(arr, num, den, B, rng), [2.5, 97.5])
    return p, lo, hi, f"{100 * p:5.2f}% [{100 * lo:4.2f}-{100 * hi:4.2f}]"


def diff_ci(a, b, num, den="n", B=1000):
    import numpy as np

    rng = np.random.default_rng(1)
    d = boot(a, num, den, B, rng) - boot(b, num, den, B, rng)
    return np.percentile(d, [2.5, 97.5])


def cosine(a, b):
    keys = set(a) | set(b)
    num = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    den = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return num / den if den else float("nan")


def cmd_report(args):
    import numpy as np
    from scipy.stats import chi2_contingency, fisher_exact

    models = {}
    for spec in args.model:
        name, _, dom = spec.partition("=")
        models[name] = set(dom.split("+"))
    pools = {  # label -> (family, kind, sets)
        "tts held-out": ("tts", "held-out", ("tts_val", "tts_test")),
        "emo held-out": ("emo", "held-out", ("emo_test",)),
        "tts train sample": ("tts", "train", ("tts_train",)),
        "emo train sample": ("emo", "train", ("emo_train",)),
    }
    clips = {}
    for m in models:
        data = json.load(open(f"{args.prefix}{m}.json"))
        for label, (_, _, sets) in pools.items():
            sub = [c for c in data if c["set"] in sets]
            if sub:
                clips[(m, label)] = sub
    arrs = {k: arr_of(v) for k, v in clips.items()}

    print("=" * 110)
    print("1. Share of syllables with a wrong onset / rime / tone (95% clip bootstrap interval); merger = errors that differ only by a")
    print("   merger pair (tone 3/6, 4/6, 2/5; onset n/l, ng/-, gw/g)")
    print(f"{'model':10} {'dataset':17} {'status':22} {'syll':>6}  {'SER':>20} {'onset':>20} {'rime':>20} {'tone':>20} merger")
    for m, dom in models.items():
        for label, (fam, kind, _) in pools.items():
            if (m, label) not in arrs:
                continue
            a = arrs[(m, label)]
            status = ("train data" if kind == "train" else "held-out, seen domain") if fam in dom else "UNSEEN domain"
            print(f"{m:10} {label:17} {status:22} {a[:, 0].sum():>6}  {fmt(a, 'err')[3]:>20} {fmt(a, 'onset')[3]:>20} "
                  f"{fmt(a, 'rime')[3]:>20} {fmt(a, 'tone')[3]:>20} {100 * a[:, KI['merger']].sum() / max(a[:, KI['err']].sum(), 1):4.0f}%")
        print()

    print("=" * 110)
    print("2. Train sample vs held-out of the same dataset (a gap = memorisation of that dataset; difference = held-out minus train, points of SER)")
    for m, dom in models.items():
        for fam in ("tts", "emo"):
            tr, ho = arrs.get((m, f"{fam} train sample")), arrs.get((m, f"{fam} held-out"))
            if tr is None or ho is None or fam not in dom:
                continue
            row = []
            for key in ("err", "onset", "rime", "tone"):
                lo, hi = diff_ci(ho, tr, key)
                pt = ho[:, KI[key]].sum() / ho[:, 0].sum() - tr[:, KI[key]].sum() / tr[:, 0].sum()
                row.append(f"{'SER' if key == 'err' else key} {100 * pt:+5.2f} [{100 * lo:+5.2f}, {100 * hi:+5.2f}]")
            print(f"  {m:10} {fam:4} " + "   ".join(row))

    print("\n" + "=" * 110)
    print("3. Do the two datasets fail in the same way? (held-out tts vs held-out emo, per model)")
    rng = np.random.default_rng(0)
    for m in models:
        if (m, "tts held-out") not in arrs or (m, "emo held-out") not in arrs:
            continue
        a, b = arrs[(m, "tts held-out")], arrs[(m, "emo held-out")]
        print(f"\n  [{m}]")
        for key in ("err", "onset", "rime", "tone"):
            lo, hi = diff_ci(a, b, key)
            pa, pb = a[:, KI[key]].sum() / a[:, 0].sum(), b[:, KI[key]].sum() / b[:, 0].sum()
            verdict = "differs" if lo > 0 or hi < 0 else "consistent (interval includes 0)"
            print(f"     {'SER' if key == 'err' else key:6} tts {100 * pa:5.2f}%  emo {100 * pb:5.2f}%  tts-emo {100 * (pa - pb):+5.2f} [{100 * lo:+5.2f}, {100 * hi:+5.2f}] {verdict}")
        tab = [[a[:, KI[k]].sum() for k in ("o1", "r1", "t1")], [b[:, KI[k]].sum() for k in ("o1", "r1", "t1")]]
        chi = chi2_contingency(tab)
        print(f"     which part fails (single-part substitutions onset/rime/tone): tts {tab[0]}, emo {tab[1]}; chi-square p = {chi[1]:.3f}"
              f" ({'different mix' if chi[1] < 0.05 else 'same mix'})")
        ta, tb = token_tables(clips[(m, "tts held-out")]), token_tables(clips[(m, "emo held-out")])
        for part in PARTS:
            nA, eA, cA = ta[0][part], ta[1][part], ta[2][part]
            nB, eB, cB = tb[0][part], tb[1][part], tb[2][part]
            shared = sum(v for k, v in cA.items() if k in cB) / max(sum(cA.values()), 1)
            shared_b = sum(v for k, v in cB.items() if k in cA) / max(sum(cB.values()), 1)
            diffs = []
            for tok in set(nA) & set(nB):
                if nA[tok] >= 30 and nB[tok] >= 30 and eA[tok] + eB[tok] >= 5:
                    p = fisher_exact([[eA[tok], nA[tok] - eA[tok]], [eB[tok], nB[tok] - eB[tok]]])[1]
                    if p < 0.01:
                        diffs.append((p, tok, eA[tok], nA[tok], eB[tok], nB[tok]))
            diffs.sort()
            print(f"     {part:5} confusion pairs: cosine similarity {cosine(cA, cB):.2f}; errors whose gold->pred pair also occurs in the other set: tts {100 * shared:.0f}%, emo {100 * shared_b:.0f}%")
            print(f"           top tts: " + ", ".join(f"{g}>{p}:{c}" for (g, p), c in cA.most_common(5)))
            print(f"           top emo: " + ", ".join(f"{g}>{p}:{c}" for (g, p), c in cB.most_common(5)))
            if diffs:
                print("           tokens with clearly different error rates (Fisher p<0.01): " + ", ".join(
                    f"{t} tts {ea}/{na} vs emo {eb}/{nb}" for _, t, ea, na, eb, nb in diffs[:6]))

    print("\n" + "=" * 110)
    print("4. How each part degrades when the model has NOT seen the dataset (rate on the unseen dataset / best seen-domain model's rate)")
    for label, (fam, kind, _) in pools.items():
        if kind != "held-out":
            continue
        seen = [m for m, dom in models.items() if fam in dom and (m, label) in arrs]
        unseen = [m for m, dom in models.items() if fam not in dom and (m, label) in arrs]
        if not seen or not unseen:
            continue
        for u in unseen:
            for s in seen:
                row = []
                for key in ("err", "onset", "rime", "tone"):
                    ru = arrs[(u, label)][:, KI[key]].sum() / arrs[(u, label)][:, 0].sum()
                    rs = arrs[(s, label)][:, KI[key]].sum() / arrs[(s, label)][:, 0].sum()
                    row.append(f"{'SER' if key == 'err' else key} {100 * ru:5.2f}% vs {100 * rs:5.2f}% = x{ru / rs if rs else float('inf'):4.1f}")
                print(f"  {label:13} unseen-domain {u:10} vs {s:10}: " + "   ".join(row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--kind", choices=["ours", "published"], required=True)
    d.add_argument("--model_id", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--sets", default=",".join(ALL_SETS))
    d.add_argument("--n_train", type=int, default=1000, help="clips sampled from each training split")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    d.add_argument("--threads", type=int, default=3)
    d.add_argument("--limit", type=int, default=0, help="decode only the first N clips of each set (smoke test)")
    d.add_argument("--out_prefix", default=os.path.join(REPO, "tmp", "preds_"))
    r = sub.add_parser("report")
    r.add_argument("--model", action="append", required=True, metavar="NAME=tts+emo")
    r.add_argument("--prefix", default=os.path.join(REPO, "tmp", "preds_"))
    args = ap.parse_args()
    cmd_decode(args) if args.cmd == "decode" else cmd_report(args)


if __name__ == "__main__":
    main()
