"""Batch synthesis with the JyutVoice TTS (frankenstein-matcha2) from jyutping input, for synthetic-data experiments.

    PYTHONPATH=<scratch packages>:<tts repo> python synth_tts.py --tts_dir <tts repo> \\
        --manifest in.json --ref_audio a.wav [--ref_audio b.wav ...] --out_dir tmp/synth [--limit N]

in.json: a list of {"id": ..., "text": "<chinese>", "phone": "<toned jyutping with spaces>"}.
Writes <out_dir>/<id>__<ref name>.wav (16 kHz mono) and <out_dir>/manifest.json with the labels used.

The TTS repo's Cython alignment module (training only) was built against another numpy; it is stubbed here, which is
enough for synthesise().
"""

import argparse
import json
import os
import sys
import time
import types


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref_audio", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--round_robin", action="store_true", help="one voice per sentence (sentence i uses reference i mod n) instead of every voice for every sentence")
    ap.add_argument("--n_timesteps", type=int, default=10)
    ap.add_argument("--length_scale", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true", help="autocast the TTS model and vocoder to bfloat16")
    ap.add_argument("--shard", default="0/1", help="i/n: synthesize every n-th sentence starting at i (run n processes in parallel)")
    ap.add_argument("--profile", action="store_true", help="print the time spent in each stage")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tts_checkpoint", default="pretrained_models/20260102_encoder_only.ckpt")
    ap.add_argument("--flow_decoder_checkpoint", default="pretrained_models/flow_decoder.pt")
    ap.add_argument("--campplus", default="pretrained_models/campplus.onnx")
    ap.add_argument("--hift", default="pretrained_models/hift.pt")
    args = ap.parse_args()
    args.manifest, args.out_dir = os.path.abspath(args.manifest), os.path.abspath(args.out_dir)
    args.ref_audio = [os.path.abspath(p) for p in args.ref_audio]
    args.tts_dir = os.path.abspath(args.tts_dir)

    os.chdir(args.tts_dir)
    sys.path.insert(0, args.tts_dir)
    # stub the compiled alignment module (training only)
    core = types.ModuleType("jyutvoice.utils.monotonic_align.core")

    def maximum_path_c(*a, **k):
        raise NotImplementedError("monotonic alignment is a training-time operation")

    core.maximum_path_c = maximum_path_c
    sys.modules["jyutvoice.utils.monotonic_align.core"] = core

    import numpy as np
    import onnxruntime
    import soundfile as sf
    import torch
    import torchaudio
    from hyperpyyaml import load_hyperpyyaml

    import infer as I                                            # the repo's inference helpers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        configs = load_hyperpyyaml(f)
    tts = configs["tts"]
    ckpt = torch.load(args.tts_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if not any("decoder" in k for k in state):
        state.update(torch.load(args.flow_decoder_checkpoint, map_location="cpu"))
    tts.load_state_dict(state)
    tts = tts.eval().to(device)
    hift = configs["hift"].eval().to(device)
    hift.load_state_dict(torch.load(args.hift, map_location="cpu"))
    opt = onnxruntime.SessionOptions()
    opt.intra_op_num_threads = 1
    spk_model = onnxruntime.InferenceSession(args.campplus, sess_options=opt, providers=["CPUExecutionProvider"])

    refs = []
    for p in args.ref_audio:
        wav, sr = torchaudio.load(p)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        refs.append((os.path.splitext(os.path.basename(p))[0], I.extract_spk_embedding(spk_model, wav).to(device)))
    print(f"{len(refs)} reference voices: {[r[0] for r in refs]}", flush=True)

    items = json.load(open(args.manifest))
    if args.limit:
        items = items[: args.limit]
    si, sn = (int(v) for v in args.shard.split("/"))
    items = items[si::sn]
    os.makedirs(args.out_dir, exist_ok=True)
    resample = torchaudio.transforms.Resample(24000, 16000)
    out, t0 = [], time.time()
    tm = {"text": 0.0, "acoustic": 0.0, "vocoder": 0.0, "post+write": 0.0}

    def tick(key, t):
        if args.profile and device.type == "cuda":
            torch.cuda.synchronize()
        tm[key] += time.time() - t
        return time.time()

    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda")
    import re

    skipped = []
    for n, it in enumerate(items):
        t = time.time()
        # one convention for every source: the characters + "。" and the toned syllables + "."; the TTS front end needs
        # exactly one syllable per character, and mid-sentence punctuation differs between the corpora
        cjk = [c for c in it["text"] if "\u4e00" <= c <= "\u9fff"]
        syl = [s for s in it["phone"].split() if re.fullmatch(r"[a-z]+[1-6]", s)]
        try:
            if len(cjk) != len(syl):
                raise ValueError(f"{len(cjk)} characters but {len(syl)} syllables")
            it = dict(it, text="".join(cjk) + "。", phone=" ".join(syl) + " .")
            text = I.word_seg(it["text"])
            x, x_len, tones, word_pos, syl_pos, lang_ids = I.get_text(text, "yue", it["phone"])
        except Exception as e:  # noqa: BLE001
            skipped.append({"id": it["id"], "why": str(e)[:120]})
            continue
        t = tick("text", t)
        for rname, spk in ([refs[(n * sn + si) % len(refs)]] if args.round_robin else refs):
            with torch.no_grad(), ac:
                res = tts.synthesise(x=x.to(device), x_lengths=x_len.to(device), lang=lang_ids.to(device), tone=tones.to(device),
                                     word_pos=word_pos.to(device), syllable_pos=syl_pos.to(device), spk_embed=spk,
                                     n_timesteps=args.n_timesteps, length_scale=args.length_scale)
                t = tick("acoustic", t)
                wav, _ = hift.inference(res["mel"])
                t = tick("vocoder", t)
            y = resample(wav.cpu().float()).squeeze(0).numpy()
            y = np.clip(y / max(np.abs(y).max(), 1e-6) * 0.9, -1, 1)
            fname = f"{it['id']}__{rname}.wav"
            sf.write(os.path.join(args.out_dir, fname), y.astype(np.float32), 16000)
            out.append({"file": fname, "id": it["id"], "ref": rname, "text": it["text"], "phone": it["phone"], "dur": len(y) / 16000})
            t = tick("post+write", t)
        if n % 25 == 0:
            print(f"{n + 1}/{len(items)} sentences, {time.time() - t0:.0f}s", flush=True)
    if skipped:
        json.dump(skipped, open(os.path.join(args.out_dir, f"skipped.{si}of{sn}.json"), "w"), ensure_ascii=False)
        print(f"skipped {len(skipped)} sentences the TTS front end rejected", flush=True)
    mname = "manifest.json" if sn == 1 else f"manifest.{si}of{sn}.json"
    json.dump(out, open(os.path.join(args.out_dir, mname), "w"), ensure_ascii=False, indent=0)
    el = time.time() - t0
    print(f"done: {len(out)} clips in {el:.0f}s ({el / max(len(out), 1):.2f} s/clip) -> {args.out_dir}", flush=True)
    if args.profile:
        print("time per stage (s per clip): " + ", ".join(f"{k} {v / max(len(out), 1):.3f}" for k, v in tm.items()), flush=True)


if __name__ == "__main__":
    main()
