import torch
from a2_3 import model, factor

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model, (x, factor),
                      "onnx/srcnn3.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
