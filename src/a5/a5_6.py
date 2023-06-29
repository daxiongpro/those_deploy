
import onnx

# 子模型提取
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/partial_model.onnx', ['22'], ['28'])
# 添加额外输出
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/submodel_1.onnx', ['22'], ['27', '31'])
# 添加冗余输入
onnx.utils.extract_model('onnxwhole_model.onnx',
                         'onnx/submodel_2.onnx', ['22', 'input.1'], ['28'])
# 输入信息不足 error
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/submodel_3.onnx', ['24'], ['28'])
# 输出 ONNX 中间节点的值
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/more_output_model.onnx', ['input.1'], ['31', '23', '25', '27'])
# 把原模型拆分成多个互不相交的子模型
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/debug_model_1.onnx', ['input.1'], ['23'])
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/debug_model_2.onnx', ['23'], ['25'])
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/debug_model_3.onnx', ['23'], ['27'])
onnx.utils.extract_model('onnx/whole_model.onnx',
                         'onnx/debug_model_4.onnx', ['25', '27'], ['31'])
