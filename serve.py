"""CPU ONNX serving for wav2vec2-bert Jyutping: 16 kHz .wav in -> phonemes out.

    POST /phoneme   (multipart file=<audio.wav>)  ->  {"phonemes": "m aa4 m aa1 ..."}
    GET  /health

Runs the heavy encoder+CTC head via ONNX Runtime (CPU); feature extraction and
CTC decoding use the checkpoint's Wav2Vec2BertProcessor (numpy, no torch).
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import Wav2Vec2BertProcessor

# fp32 is both faster and more accurate than dynamic-INT8 for this conformer on
# CPU (INT8 measured ~2x slower with a tiny accuracy drop). Override with
# MODEL_ONNX=onnx/run22/model.int8.onnx only if you need the smaller ~600 MB footprint.
MODEL_ONNX = os.environ.get("MODEL_ONNX", "onnx/run22/model.onnx")
PROCESSOR_DIR = os.environ.get("PROCESSOR_DIR", "checkpoints/run22_8000")
TARGET_SR = 16000

_NP = {"tensor(int32)": np.int32, "tensor(int64)": np.int64,
       "tensor(float)": np.float32, "tensor(float16)": np.float16}

app = FastAPI(title="wav2vec2bert-jyutping (ONNX/CPU)", version="1")


@app.on_event("startup")
def _load():
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))  # 0 = all cores
    app.state.sess = ort.InferenceSession(MODEL_ONNX, sess_options=so,
                                          providers=["CPUExecutionProvider"])
    app.state.in_dtype = {i.name: _NP.get(i.type, np.float32) for i in app.state.sess.get_inputs()}
    app.state.processor = Wav2Vec2BertProcessor.from_pretrained(PROCESSOR_DIR)


def _read_wav(raw: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:                      # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:                     # resample to 16 kHz if needed
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_ONNX)}


@app.post("/phoneme")
async def phoneme(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        audio = _read_wav(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not decode audio: {e}")

    proc = app.state.processor
    inputs = proc(audio, sampling_rate=TARGET_SR, return_tensors="np")
    feed = {
        "input_features": inputs["input_features"].astype(app.state.in_dtype["input_features"]),
        "attention_mask": inputs["attention_mask"].astype(app.state.in_dtype["attention_mask"]),
    }
    t0 = time.time()
    logits = app.state.sess.run(["logits"], feed)[0]
    dt = time.time() - t0

    pred_ids = logits.argmax(-1)[0]
    phonemes = proc.decode(pred_ids)
    return {
        "phonemes": phonemes,
        "audio_seconds": round(len(audio) / TARGET_SR, 3),
        "infer_seconds": round(dt, 3),
        "rtf": round(dt / max(len(audio) / TARGET_SR, 1e-6), 3),
    }
