"""Compose a reference wav for listening checks: the dictionary recording of every syllable of a sample, joined in order.

    python compose_reference.py --dir tmp/synth_samples_rare [--gap 0.12]

Reads README.txt in --dir (lines "<clip>.wav: <text>" followed by the toned jyutping), fetches each syllable from
https://cantonesepinyin.com/api/audio/<syllable> (cached in tmp/ref_syllables), trims the silence around it, matches the
loudness, and writes REF_<clip>.wav (the syllables only) and COMPARE_<clip>.wav (the TTS clip, a pause, then the reference).
Single-syllable recordings are read in isolation, so tones and durations differ from connected speech: compare the
syllable identity and its tone category, not the rhythm.
"""

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(REPO, "tmp", "ref_syllables")
SR = 16000


def fetch(syl):
    path = os.path.join(CACHE, f"{syl}.mp3")
    if not os.path.exists(path):
        req = urllib.request.Request(f"https://cantonesepinyin.com/api/audio/{syl}", headers={"User-Agent": "Mozilla/5.0 (listening check)"})
        try:
            data = urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            return None if e.code == 404 else (_ for _ in ()).throw(e)
        if len(data) < 500:
            return None
        open(path, "wb").write(data)
        time.sleep(0.3)                                              # be polite to the service
    return path


def load(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(out, dtype=np.float32).copy()
    env = np.abs(y)
    idx = np.nonzero(env > 0.03 * env.max())[0]
    y = y[max(idx[0] - int(0.02 * SR), 0): idx[-1] + int(0.03 * SR)] if len(idx) else y
    return y / (np.sqrt((y ** 2).mean()) + 1e-9) * 0.08               # equal RMS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gap", type=float, default=0.12)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    txt = open(os.path.join(args.dir, "README.txt"), encoding="utf-8").read()
    items = re.findall(r"^(\S+\.wav): (.*)\n\s+((?:[a-z]+[1-6]\s*)+\.?)\s*$", txt, re.M)
    missing = set()
    for clip, text, phone in items:
        syl = phone.split()
        parts, gap = [], np.zeros(int(args.gap * SR), dtype=np.float32)
        for s in syl:
            p = fetch(s)
            if p is None:
                missing.add(s)
                parts += [np.zeros(int(0.4 * SR), dtype=np.float32), gap]            # silence where the recording is missing
            else:
                parts += [load(p), gap]
        ref = np.concatenate(parts)
        sf.write(os.path.join(args.dir, f"REF_{clip}"), ref, SR)
        tts, sr = sf.read(os.path.join(args.dir, clip), dtype="float32")
        if tts.ndim > 1:
            tts = tts.mean(1)
        if sr != SR:
            import librosa
            tts = librosa.resample(tts, orig_sr=sr, target_sr=SR)
        sf.write(os.path.join(args.dir, f"COMPARE_{clip}"), np.concatenate([tts, np.zeros(int(0.8 * SR), dtype=np.float32), ref]), SR)
        print(f"{clip}: TTS {len(tts) / SR:.1f} s | reference {len(ref) / SR:.1f} s | {' '.join(syl)}")
    print(f"{len(items)} clips composed; syllables with no recording on the service: {sorted(missing) or 'none'}")


if __name__ == "__main__":
    main()
