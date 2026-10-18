"""Export checkpoints/run22_8000 (Wav2Vec2BertForJyutpingCTC) to ONNX for
CPU serving, then INT8-quantize and verify parity with the PyTorch model.

    python export_onnx.py --ckpt checkpoints/run22_8000 --out onnx/run22

Produces:
    onnx/run22/model.onnx        fp32 graph (input_features, attention_mask -> logits)
    onnx/run22/model.int8.onnx   dynamic INT8 quantized (smaller/faster on CPU)

The graph emits raw CTC logits; decoding (argmax + collapse) is done in Python
at serve time with the checkpoint's Wav2Vec2CTCTokenizer.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2BertProcessor

from model import Wav2Vec2BertForJyutpingCTC


class LogitsOnly(nn.Module):
    """Wrap the model so ONNX sees a clean (features, mask) -> logits graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, attention_mask):
        return self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/run22_8000")
    ap.add_argument("--out", default="onnx/run22")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dummy-seconds", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fp32_path = os.path.join(args.out, "model.onnx")
    int8_path = os.path.join(args.out, "model.int8.onnx")

    print(f"[load] processor + model from {args.ckpt}")
    processor = Wav2Vec2BertProcessor.from_pretrained(args.ckpt)
    # eager attention exports cleanly (SDPA/flash paths often don't).
    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        args.ckpt, attn_implementation="eager", torch_dtype=torch.float32
    ).eval()
    wrapper = LogitsOnly(model).eval()

    # Dummy 16 kHz audio -> processor features (shapes drive the export).
    dummy = np.zeros(int(16000 * args.dummy_seconds), dtype=np.float32)
    inputs = processor(dummy, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"].to(torch.float32)
    mask = inputs["attention_mask"]
    print(f"[shapes] input_features={tuple(feats.shape)} attention_mask={tuple(mask.shape)}")

    print(f"[export] -> {fp32_path} (opset {args.opset})")
    torch.onnx.export(
        wrapper,
        (feats, mask),
        fp32_path,
        input_names=["input_features", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "time"},
            "attention_mask": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "out_time"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Parity check: torch vs onnxruntime on the same dummy input.
    import onnxruntime as ort

    with torch.no_grad():
        ref = wrapper(feats, mask).numpy()
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])

    # Feed each input as the exact dtype the graph declares (the exporter may
    # pick int32 for the mask, float32 for features, etc.).
    _np_dtype = {"tensor(int32)": np.int32, "tensor(int64)": np.int64,
                 "tensor(float)": np.float32, "tensor(float16)": np.float16}
    want = {i.name: _np_dtype.get(i.type, np.float32) for i in sess.get_inputs()}
    feed = {
        "input_features": feats.numpy().astype(want["input_features"]),
        "attention_mask": mask.numpy().astype(want["attention_mask"]),
    }
    print(f"[onnx inputs] {[(i.name, i.type) for i in sess.get_inputs()]}")
    onx = sess.run(["logits"], feed)[0]
    diff = float(np.abs(ref - onx).max())
    argmax_match = bool((ref.argmax(-1) == onx.argmax(-1)).all())
    print(f"[parity] max_abs_diff={diff:.3e}  argmax_identical={argmax_match}  logits_shape={onx.shape}")

    # INT8 dynamic quantization for CPU.
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        print(f"[quantize] -> {int8_path} (dynamic INT8)")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        s32 = os.path.getsize(fp32_path) / 1e6
        s8 = os.path.getsize(int8_path) / 1e6
        print(f"[size] fp32={s32:.0f} MB  int8={s8:.0f} MB")
    except Exception as e:  # noqa: BLE001
        print(f"[quantize] skipped: {e}")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
