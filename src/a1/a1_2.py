import torch
from a1_1 import model

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "onnx/srcnn.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
