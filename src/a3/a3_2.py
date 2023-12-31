import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        return x


model = Model()
dummy_input = torch.rand(1, 3, 10, 10)
model_names = ['onnx/model_static.onnx',
               'onnx/model_dynamic_0.onnx',
               'onnx/model_dynamic_23.onnx']

dynamic_axes_0 = {
    'in': [0],
    'out': [0]
}
dynamic_axes_23 = {
    'in': [2, 3],
    'out': [2, 3]
}

torch.onnx.export(model, dummy_input, model_names[0],
                  input_names=['in'], output_names=['out'])
torch.onnx.export(model, dummy_input, model_names[1],
                  input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_0)
torch.onnx.export(model, dummy_input, model_names[2],
                  input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_23)
