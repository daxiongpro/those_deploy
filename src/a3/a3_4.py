import torch

"""
我们希望模型在导出至 ONNX 时有一些不同的行为模型在直接用 PyTorch 推理时有一套逻辑，
而在导出的ONNX模型中有另一套逻辑。
比如，我们可以把一些后处理的逻辑放在模型里，以简化除运行模型之外的其他代码。
但是，这些代码对只关心模型训练的开发者和用户来说很不友好，突兀的部署逻辑会降低代码整体的可读性。
同时，is_in_onnx_export只能在每个需要添加部署逻辑的地方都“打补丁”，难以进行统一的管理。
"""


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        if torch.onnx.is_in_onnx_export():
            x = torch.clip(x, 0, 1)
        return x
