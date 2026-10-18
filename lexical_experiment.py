"""Can lexical context (the characters being said) resolve the tone-merger errors?  (Mok & Wong 2010: listeners resolve
T2/T5, T3/T6, T4/T6 with context.)

    pip install ToJyutping            # or --target <dir> and pass --extra_path <dir>
    python lexical_experiment.py --extra_path <dir>      # needs tmp/preds_run21.json (error_consistency.py decode) and
                                                         # tmp/pitch_feats.json (check_tone_pitch.py build: tone posteriors)

Part A - ceiling. A "reader" that knows the TRUE characters and gives each one the syllable the labellers most often gave it
(learned from the training transcripts, optionally with its neighbours as context) is scored against the labels. This is how
much of the label the text alone determines.

Part B - lexical decoder. The characters are latent. For every predicted syllable the candidates are the six tones of the same
rime (scored with the model's own tone log-posterior at the rime's peak frame, weight beta) times the characters that the
labellers read with that syllable; a character 4-gram language model trained on the training text scores the character
sequence; a beam search picks the best (tone, character) sequence and the output syllable is the chosen tone's syllable.
beta is tuned on the validation clips only.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from error_consistency import TONE_MERGERS, align, parse  # noqa: E402

CJK = re.compile(r"[一-鿿]")
SYL = re.compile(r"[a-z]+[1-6]")


class CharKN:
    """interpolated Kneser-Ney character n-gram"""

    def __init__(self, sents, order=4, D=0.75):
        self.order, self.D = order, D
        raw = Counter()
        for s in sents:
            t = ["<s>"] * (order - 1) + s + ["</s>"]
            for i in range(order - 1, len(t)):
                raw[tuple(t[i - order + 1: i + 1])] += 1
        self.cnt = {order: raw}
        for n in range(order - 1, 0, -1):
            c = Counter()
            for ng in self.cnt[n + 1]:
                c[ng[1:]] += 1
            self.cnt[n] = c
        self.tot, self.types = {}, {}
        for n in range(1, order + 1):
            tot, ty = Counter(), Counter()
            for ng, c in self.cnt[n].items():
                tot[ng[:-1]] += c
                ty[ng[:-1]] += 1
            self.tot[n], self.types[n] = tot, ty
        self.V = len(self.cnt[1]) + 1
        self.memo = {}

    def logp(self, w, ctx):
        ctx = ("<s>",) * (self.order - 1 - len(ctx)) + tuple(ctx)
        key = (ctx, w)
        v = self.memo.get(key)
        if v is None:
            p = 1.0 / self.V
            for n in range(1, self.order + 1):
                c = ctx[len(ctx) - (n - 1):] if n > 1 else ()
                tot = self.tot[n].get(c, 0)
                if tot:
                    p = max(self.cnt[n].get(c + (w,), 0) - self.D, 0.0) / tot + self.D * self.types[n][c] / tot * p
            v = self.memo[key] = math.log(p)
        return v


def char_syllable_pairs(tj, text, label_syls):
    """[(char, label syllable)] when every character of the text has one syllable and the counts match the label, else None"""
    items = [(ch, SYL.findall(jp)) for ch, jp in tj.get_jyutping_list(text) if jp]
    if any(len(s) != 1 for _, s in items) or len(items) != len(label_syls):
        return None
    return [(ch, s) for (ch, _), s in zip(items, label_syls)]


def build_lexicon(tj, emo_path):
    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text

    tts = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select_columns(["text", "phone"])
    emo = load_any_dataset(emo_path)
    emo = emo["train"] if hasattr(emo, "keys") else emo
    emo = emo.select_columns(["text", "labels"])
    sents, r1, r_next, r_prev, r_both = [], defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    used = total = 0
    for texts, labels in ((tts["text"], tts["phone"]), (emo["text"], emo["labels"])):
        for t, l in zip(texts, labels):
            total += 1
            sents.append([c for c in t if CJK.search(c)])
            pairs = char_syllable_pairs(tj, t, normalize_jyutping_text(l).split())
            if pairs is None:
                continue
            used += 1
            chars = [c for c, _ in pairs]
            for i, (c, s) in enumerate(pairs):
                p = chars[i - 1] if i else "<s>"
                n = chars[i + 1] if i + 1 < len(chars) else "</s>"
                r1[c][s] += 1; r_prev[(p, c)][s] += 1; r_next[(c, n)][s] += 1; r_both[(p, c, n)][s] += 1
    print(f"lexicon from {used:,} of {total:,} training sentences ({len(r1):,} characters)", flush=True)
    return sents, r1, r_prev, r_next, r_both


def read_chars(chars, r1, r_prev, r_next, r_both, context):
    out = []
    for i, c in enumerate(chars):
        p = chars[i - 1] if i else "<s>"
        n = chars[i + 1] if i + 1 < len(chars) else "</s>"
        cnt = None
        if context:
            for table, key, need in ((r_both, (p, c, n), 2), (r_next, (c, n), 3), (r_prev, (p, c), 3)):
                x = table.get(key)
                if x and sum(x.values()) >= need:
                    cnt = x
                    break
        cnt = cnt or r1.get(c)
        out.append(cnt.most_common(1)[0][0] if cnt else "?")
    return out


# ----------------------------------------------------------------------------- part B worker
G = {}


def decode_clip(args):
    pred, tp, beta = args
    I, lm, B, K = G["I"], G["lm"], G["B"], G["K"]
    cands = []
    for s, post in zip(pred, tp):
        opts = []
        for t, lpv in post.items():
            v = s[:-1] + t
            for c in I.get(v, ())[:K]:
                opts.append((v, c, beta * lpv))
        cands.append(opts or [(s, "<unk>", 0.0)])
    beams = [(0.0, (), ())]
    for opts in cands:
        new = []
        for sc, ch, sy in beams:
            ctx = ch[-3:]
            for v, c, ac in opts:
                new.append((sc + ac + lm.logp(c, ctx), ch + (c,), sy + (v,)))
        new.sort(key=lambda x: -x[0])
        beams = new[:B]
    best = max(beams, key=lambda x: x[0] + lm.logp("</s>", x[1][-3:]))
    return list(best[2])


def merger_only(r, h):
    po, pn, pc, pt = parse(r)
    ho, hn, hc, ht = parse(h)
    return (po, pn, pc) == (ho, hn, hc) and pt != ht and frozenset((pt, ht)) in TONE_MERGERS


def errors(gold, pred):
    e = m = 0
    for op, _, r, h in align(gold, pred):
        if op != "=":
            e += 1
            m += op == "S" and merger_only(r, h)
    return e, m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra_path", default=None)
    ap.add_argument("--emo_path", default="/home/joseph/data/dataset_emo_jyutping")
    ap.add_argument("--preds", default=os.path.join(REPO, "tmp", "preds_run21.json"))
    ap.add_argument("--feats", default=os.path.join(REPO, "tmp", "pitch_feats.json"))
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--chars_per_syllable", type=int, default=12)
    ap.add_argument("--procs", type=int, default=11)
    args = ap.parse_args()
    if args.extra_path:
        sys.path.insert(0, args.extra_path)
    import ToJyutping as tj

    from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
    from jyutping import normalize_jyutping_text

    sents, r1, r_prev, r_next, r_both = build_lexicon(tj, args.emo_path)
    train_set = {tuple(s) for s in sents}
    lm = CharKN(sents)
    print(f"character LM: {len(lm.cnt[4]):,} 4-grams, vocabulary {lm.V:,}", flush=True)
    inv = defaultdict(list)                       # syllable -> characters the labellers read with it, most frequent first
    for c, cnt in r1.items():
        for s, n in cnt.items():
            if n >= 2:
                inv[s].append((n * 1.0, c))
    I = {s: [c for _, c in sorted(v, reverse=True)] for s, v in inv.items()}

    sp = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)
    emo = load_any_dataset(args.emo_path)["test"]
    texts = {"tts_val": sp["validation"]["text"], "tts_test": sp["test"]["text"], "emo_test": emo["text"]}
    preds = {(c["set"], c["idx"]): c for c in json.load(open(args.preds)) if c["set"] in texts}
    feats = json.load(open(args.feats))["clips"]
    clips = []
    for (s, i), c in sorted(preds.items()):
        f = feats.get(f"{s}:{i}")
        if f is None or len(f["tp"]) != len(c["pred"]):
            continue
        chars = [ch for ch in texts[s][i] if CJK.search(ch)]
        clips.append({"set": s, "idx": i, "gold": c["gold"], "pred": c["pred"], "tp": f["tp"], "chars": chars,
                      "seen": tuple(chars) in train_set})
    print(f"held-out clips usable: {len(clips)} ({Counter(c['set'] for c in clips)})\n", flush=True)
    sets = ("tts_val", "tts_test", "emo_test")

    print("=" * 100)
    print("A. How much of the label does the text alone determine? (true characters -> syllables the labellers most often gave them)")
    print(f"{'':28} " + " ".join(f"{s:>20}" for s in sets))
    base = {s: sum(errors(c["gold"], c["pred"])[0] for c in clips if c["set"] == s) for s in sets}
    nsyl = {s: sum(len(c["gold"]) for c in clips if c["set"] == s) for s in sets}
    print(f"{'ASR model (run21)':28} " + " ".join(f"{base[s]:>8} ({100 * base[s] / nsyl[s]:4.2f}%)".rjust(20) for s in sets))
    for name, ctx in (("reader, character only", False), ("reader, with neighbours", True)):
        e = {s: sum(errors(c["gold"], read_chars(c["chars"], r1, r_prev, r_next, r_both, ctx))[0] for c in clips if c["set"] == s) for s in sets}
        print(f"{name:28} " + " ".join(f"{e[s]:>8} ({100 * e[s] / nsyl[s]:4.2f}%)".rjust(20) for s in sets))
    print("   (ToJyutping alone: 3.0% of tts syllables and 5.0% of emo syllables differ from the label)")

    print("\n" + "=" * 100)
    print("B. Lexical decoder: model tone posterior (weight beta) + character 4-gram LM, characters latent")
    G.update(I=I, lm=lm, B=args.beam, K=args.chars_per_syllable)
    ctx = mp.get_context("fork")
    results = {}
    with ctx.Pool(args.procs) as pool:
        for beta in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            outs = pool.map(decode_clip, [(c["pred"], c["tp"], beta) for c in clips], chunksize=8)
            results[beta] = outs
            print(f"   beta {beta} done", flush=True)
    print(f"\n{'beta':>6} " + " ".join(f"{s + ' errors':>26}" for s in sets) + "     merger-pair tone errors (val / test / emo)")
    b0 = {s: [errors(c["gold"], c["pred"]) for c in clips if c["set"] == s] for s in sets}
    print(f"{'model':>6} " + " ".join(f"{sum(x[0] for x in b0[s]):>26}" for s in sets) + "     " + " / ".join(str(sum(x[1] for x in b0[s])) for s in sets))
    table = {}
    for beta, outs in results.items():
        cells, mm = [], []
        for s in sets:
            ev = [errors(c["gold"], o) for c, o in zip(clips, outs) if c["set"] == s]
            e, m = sum(x[0] for x in ev), sum(x[1] for x in ev)
            table[(beta, s)] = e
            cells.append(f"{e:>8} ({e - sum(x[0] for x in b0[s]):+d})".rjust(26)); mm.append(str(m))
        print(f"{beta:>6} " + " ".join(cells) + "     " + " / ".join(mm))
    best = min(results, key=lambda b: table[(b, "tts_val")])
    print(f"\nbeta tuned on tts_val: {best}")
    for s in ("tts_test", "emo_test"):
        fixed = broke = changed = 0
        for c, o in zip(clips, results[best]):
            if c["set"] != s or len(o) != len(c["pred"]):
                continue
            for op, i, j in [(op, i, j) for op, i, j in _idx_align(c["gold"], c["pred"]) if op in ("=", "S")]:
                new = o[j]
                changed += new != c["pred"][j]
                fixed += c["pred"][j] != c["gold"][i] and new == c["gold"][i]
                broke += c["pred"][j] == c["gold"][i] and new != c["gold"][i]
        print(f"   {s:9} syllables changed {changed}: errors fixed {fixed}, correct syllables broken {broke}, net {broke - fixed:+d}")
    for flag in (True, False):
        for s in ("tts_test", "emo_test"):
            sub = [(c, o) for c, o in zip(clips, results[best]) if c["set"] == s and c["seen"] == flag]
            if sub:
                e0 = sum(errors(c["gold"], c["pred"])[0] for c, _ in sub)
                e1 = sum(errors(c["gold"], o)[0] for c, o in sub)
                print(f"   {s:9} sentences {'seen in the LM training text' if flag else 'not in the LM training text':30} {len(sub):5} clips: {e0} -> {e1}")


def _idx_align(ref, hyp):
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


if __name__ == "__main__":
    main()
