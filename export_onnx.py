# Python script to export ONNX models

import torch
import torch.onnx
from iwpc.utils import latest_ckpt
from TMTrain_onnx import KappaLightning  
import numpy as np

ckpt = latest_ckpt("/Users/albaburgosmondejar/QMISID/shift_logs/kappa/version_11")
lit = KappaLightning.load_from_checkpoint(ckpt, map_location="cpu", strict=False)
lit.eval()

class ProbWrapper(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        x_norm = torch.stack([x[:, 0], x[:, 1]], dim=1)
        logits = self.base(x_norm)
        return logits

model = ProbWrapper(
    lit.model
).eval()
dummy = torch.randn(1, 2, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy,
    "qmisid_data.onnx",
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["prob"],
    dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
    dynamo=True,
)
