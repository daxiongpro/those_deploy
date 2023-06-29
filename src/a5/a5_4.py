import onnx
model = onnx.load('onnx/linear_func.onnx')

node = model.graph.node
node[1].op_type = 'Sub'

onnx.checker.check_model(model)
onnx.save(model, 'onnx/linear_func_2.onnx')
