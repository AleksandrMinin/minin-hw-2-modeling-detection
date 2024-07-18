import torch
import segmentation_models_pytorch as smp

from src.config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH


def test_forward(config_test):
    model = smp.Unet(encoder_name=config_test.model_kwargs["encoder_name"],
                         encoder_weights=config_test.model_kwargs["encoder_weights"],
                         classes=NUM_CLASSES,
                         aux_params={'pooling': 'avg', 'dropout': 0.2, 'classes': NUM_CLASSES})


    dummy_input = torch.ones(1, 3, IMG_HEIGHT, IMG_WIDTH)
    model_ouput = model(dummy_input)
    assert model_ouput[0].shape == (1, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH)
    assert model_ouput[1].shape == (1, NUM_CLASSES)
