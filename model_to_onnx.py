import torch
import segmentation_models_pytorch as smp
import onnx
from onnxsim import simplify

from src.config import config
from src.config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH
from src.constants import TORCH_FILE, ONNX_FILE


DEVICE = 'cpu'

state_dict = torch.load(TORCH_FILE)
model = smp.Unet(encoder_name=config.model_kwargs["encoder_name"],
                 encoder_weights=config.model_kwargs["encoder_weights"],
                 classes=NUM_CLASSES,
                 aux_params={'pooling': 'avg', 'dropout': 0.2, 'classes': NUM_CLASSES})
model.load_state_dict(state_dict)

print('---------------------------------------------------------------------------------')
print('--------------------------------START ONNX EXPORT--------------------------------')
print('---------------------------------------------------------------------------------')
dummy_input = torch.rand(1, 3, IMG_HEIGHT, IMG_WIDTH, device=DEVICE)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
)
print('---------------------------------------------------------------------------------')
print('--------------------------------FINISH ONNX EXPORT-------------------------------')
print('---------------------------------------------------------------------------------')
print('\n')
print('---------------------------------------------------------------------------------')
print('------------------------------------CHECK ONNX-----------------------------------')
print('---------------------------------------------------------------------------------')
onnx_model = onnx.load(ONNX_FILE)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
print('---------------------------------------------------------------------------------')
print('-------------------------------------ONNX OK-------------------------------------')
print('---------------------------------------------------------------------------------')
print('\n')
print('---------------------------------------------------------------------------------')
print('--------------------------------SIMPLIFY ONNX MODEL------------------------------')
print('---------------------------------------------------------------------------------')
onnx_model_simp, check = simplify(onnx_model)
onnx_model_simp, check = simplify(onnx_model)
print('---------------------------------------------------------------------------------')
if check:
    print('---------------------------------SIMPLIFY SUCCESS--------------------------------')
else:
    print('----------------------------------SIMPLIFY FAILED--------------------------------')
print('---------------------------------------------------------------------------------')
