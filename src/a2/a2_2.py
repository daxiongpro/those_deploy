import torch
from a2_1 import model

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    # torch.onnx.export(model, (x, 3),
    torch.onnx.export(model, (x, torch.tensor(3)),
                      "onnx/srcnn2.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])
