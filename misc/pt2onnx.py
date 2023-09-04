import torch
import torchvision
import onnx
from onnxsim import simplify


def py2onnx(model_name):
    # build model
    test_model = eval(f'torchvision.models.{model_name}()')
    test_model.eval()

    torch.save(test_model, f'./model/{model_name}.pt')
    # export onnx
    imgsz = 512
    dummy_input = torch.zeros(1, 3, imgsz, imgsz).float()
    torch.onnx.export(test_model,
                      dummy_input,
                      f'./model/{model_name}.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=12,
                      do_constant_folding=True)

    # export onnx_simplify
    onnx_file = f'./model/{model_name}.onnx'
    model = onnx.load(onnx_file)
    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, f'./model/{model_name}_simplify.onnx')


if __name__ == '__main__':
    model_name = 'resnet18'
    py2onnx(model_name)
