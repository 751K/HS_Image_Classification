# model_init.py
import torch.nn as nn


def get_available_models():
    return {name: globals()[name] for name in globals() if
            isinstance(globals()[name], type) and issubclass(globals()[name], nn.Module)}


AVAILABLE_MODELS = get_available_models()


def get_model(model_name, **kwargs):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(AVAILABLE_MODELS.keys())}")

    model_class = AVAILABLE_MODELS[model_name]

    # 统一参数名
    input_channels = kwargs.pop('in_channels', kwargs.pop('input_channels', None))
    num_classes = kwargs.pop('n_classes', kwargs.pop('num_classes', None))

    if input_channels is None or num_classes is None:
        raise ValueError("Both 'input_channels' and 'num_classes' must be provided.")

    # 使用统一的参数名创建模型实例
    return model_class(input_channels=input_channels, num_classes=num_classes, **kwargs)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def create_model(model_name, input_channels, num_classes):
    model = get_model(model_name, input_channels=input_channels, num_classes=num_classes)
    model.apply(init_weights)
    return model
