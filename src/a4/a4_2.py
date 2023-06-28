import onnxruntime
import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


model = Model()
input = torch.rand(1, 3, 10, 10)
torch_output = model(input).detach().numpy()

sess = onnxruntime.InferenceSession('onnx/asinh.onnx')
ort_output = sess.run(None, {'0': input.numpy()})[0]

assert np.allclose(torch_output, ort_output)  # 来保证两个结果张量的误差在一个可以允许的范围内。
