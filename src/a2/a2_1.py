import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np


class SuperResolutionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = interpolate(x,
                        # scale_factor=upscale_factor,
                        scale_factor=upscale_factor.item(),
                        mode='bicubic',
                        align_corners=False)

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet()

    state_dict = torch.load('checkpoints/srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()

input_img = cv2.imread('imgs/face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
"""
PyTorch 模型在导出到 ONNX 模型时，模型的输入参数的类型必须全部是 torch.Tensor。
而实际上我们传入的第二个参数" 3 "是一个整形变量。
这不符合 PyTorch 转 ONNX 的规定。
"""
# torch_output = model(torch.from_numpy(input_img), 3).detach().numpy()
torch_output = model(torch.from_numpy(input_img),
                     torch.tensor(3)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("imgs/face_torch_2.png", torch_output)
