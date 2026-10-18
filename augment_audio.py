"""Audio augmentation for training clips: noise, echo / reverberation, speed (+/- 5%, plain resampling), band limiting, gain.

    python augment_audio.py --in_dir tmp/synth_plan/out_first --out_dir tmp/synth_plan/aug_first [--copies 1] [--seed 0]
    python augment_audio.py --real_rows 6000 --out_dir tmp/synth_plan/aug_real6000        # augmented copies of real training clips

Reads manifest.json (file, phone, ...) from --in_dir (synth_tts.py output) or takes the first --real_rows rows of the tts20250516
TRAIN split, writes 16 kHz wavs and a manifest.json with the same labels plus the augmentation parameters. The label never
changes: only operations that keep the syllables (and their tones) intact are used. Speed uses plain resampling, not the
phase vocoder (which damages recognition for every model, see CLAUDE.md).
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def augment(y, rng, others):
    import numpy as np
    from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

    sr = 16000
    info = {}
    # 1. speed, plain resampling (also moves the pitch by the same ratio)
    f = rng.choice([0.95, 1.0, 1.05])
    if f != 1.0:
        up, down = (20, 21) if f == 1.05 else (20, 19)
        y = resample_poly(y, up, down).astype(np.float32)
        info["speed"] = float(f)
    # 2. echo / reverberation
    r = rng.random()
    if r < 0.45:
        if rng.random() < 0.5:                                           # discrete echoes
            out = y.copy()
            taps = []
            for _ in range(rng.integers(1, 3)):
                d, g = int(rng.uniform(0.04, 0.2) * sr), float(rng.uniform(0.1, 0.35))
                out[d:] += g * y[: len(y) - d]
                taps.append([round(d / sr, 3), round(g, 2)])
            y = out
            info["echo"] = taps
        else:                                                            # synthetic room impulse response
            rt60 = float(rng.uniform(0.2, 0.7))
            n = int(rt60 * sr)
            t = np.arange(n) / sr
            rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
            rir[0] = 1.0
            rir /= np.sqrt((rir ** 2).sum())
            wet = float(rng.uniform(0.2, 0.6))
            y = ((1 - wet) * y + wet * fftconvolve(y, rir)[: len(y)] * np.sqrt((y ** 2).mean() / max((fftconvolve(y, rir)[: len(y)] ** 2).mean(), 1e-9))).astype(np.float32)
            info["reverb"] = {"rt60": round(rt60, 2), "wet": round(wet, 2)}
    # 3. additive noise
    if rng.random() < 0.55:
        snr = float(rng.uniform(8, 30))
        kind = rng.choice(["white", "pink", "babble"]) if others else rng.choice(["white", "pink"])
        if kind == "babble":
            noise = np.zeros(len(y), dtype=np.float32)
            for o in rng.choice(len(others), size=3, replace=False):
                z = others[o]
                s = int(rng.integers(0, max(len(z) - len(y), 1)))
                seg = z[s: s + len(y)]
                noise[: len(seg)] += seg
        else:
            noise = rng.standard_normal(len(y)).astype(np.float32)
            if kind == "pink":
                spec = np.fft.rfft(noise)
                spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
                noise = np.fft.irfft(spec, len(y)).astype(np.float32)
        py, pn = float((y ** 2).mean()), float((noise ** 2).mean())
        y = y + noise * np.sqrt(py / (pn * 10 ** (snr / 10) + 1e-12))
        info["noise"] = {"type": str(kind), "snr_db": round(snr, 1)}
    # 4. band limiting (telephone-like)
    if rng.random() < 0.2:
        cut = float(rng.uniform(3400, 5000))
        y = sosfilt(butter(6, cut, btype="low", fs=sr, output="sos"), y).astype(np.float32)
        info["lowpass_hz"] = round(cut)
    # 5. gain, then peak limit
    g = float(10 ** (rng.uniform(-12, 3) / 20))
    y = y * g
    peak = float(np.abs(y).max())
    if peak > 0.95:
        y = y * (0.95 / peak)
    info["gain_db"] = round(20 * np.log10(g), 1)
    if len(info) <= 1:                                                  # make sure something other than gain happened
        return augment(y, rng, others)
    return y.astype(np.float32), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir")
    ap.add_argument("--real_rows", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import librosa
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(args.seed)
    if args.in_dir:
        man = json.load(open(os.path.join(args.in_dir, "manifest.json")))
        clips = [(m["file"], m["phone"], os.path.join(args.in_dir, m["file"]), m) for m in man]
        load = lambda p: librosa.load(p, sr=16000, mono=True)[0].astype(np.float32)
    else:
        from datasets import Audio

        from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
        from eval_w2v2bert_nucleus_tone import read_audio

        d = make_train_validation_test_split(load_any_dataset(DEFAULT_DATASET), seed=42)["train"].select(range(args.real_rows))
        d = d.select_columns(["audio", "phone"]).cast_column("audio", Audio(decode=False))
        clips = [(f"real{i:05d}.wav", row["phone"].strip(), row["audio"], {"id": f"real{i:05d}", "ref": "real"}) for i, row in enumerate(d)]

        def load(a):
            y, sr = read_audio(a)
            return (librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y).astype(np.float32)
    os.makedirs(args.out_dir, exist_ok=True)
    idx = rng.choice(len(clips), size=min(40, len(clips)), replace=False)
    pool = [load(clips[i][2]) for i in idx]
    out = []
    for n, (fname, phone, src, meta) in enumerate(clips):
        y = load(src)
        for c in range(args.copies):
            ya, info = augment(y, rng, pool)
            name = f"{os.path.splitext(fname)[0]}__aug{c}.wav"
            sf.write(os.path.join(args.out_dir, name), ya, 16000)
            out.append(dict(meta, file=name, phone=phone, aug=info, dur=len(ya) / 16000))
        if n % 500 == 0:
            print(f"{n}/{len(clips)}", flush=True)
    json.dump(out, open(os.path.join(args.out_dir, "manifest.json"), "w"), ensure_ascii=False)
    from collections import Counter
    kinds = Counter(k for m in out for k in m["aug"] if k != "gain_db")
    print(f"wrote {len(out)} augmented clips to {args.out_dir}; operations: {dict(kinds)}")


if __name__ == "__main__":
    main()
