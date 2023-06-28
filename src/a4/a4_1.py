from torch.onnx.symbolic_registry import register_op
import torch

"""
pytorch 算子 asinh 定义在：
/usr/miniconda3/envs/test/lib/python3.8/site-packages/torch/nn/functional.pyi
/usr/miniconda3/envs/test/lib/python3.8/site-packages/torch/_C/_VariableFunctions.pyi
"""


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


def asinh_symbolic(g, input, *, out=None):
    """
    _VariableFunctions.pyi 下找到：
    def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
    把参数一一填入。
    从除g以外的第二个输入参数开始，其输入参数应该严格对应它在 ATen 中的定义。
    """
    return g.op("Asinh", input)


register_op('asinh', asinh_symbolic, '', 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'onnx/asinh.onnx')
