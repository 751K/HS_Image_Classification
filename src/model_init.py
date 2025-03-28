# model_init.py
import torch.nn as nn
import inspect

from Train_and_Eval.device import get_device

# Notice: Do not delete the following line
from src.CNNBase.__init__ import *
from src.TransformerBase.__init__ import *
from src.MambaBase.__init__ import *


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
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.GRU)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    elif isinstance(m, (nn.Sequential, nn.ModuleList)):
        for sub_module in m.children():
            init_weights(sub_module)
    elif isinstance(m, nn.Parameter):
        if m.dim() > 1:
            nn.init.xavier_uniform_(m)


def create_model(model_name, config, logger=None):
    """
    根据模型需要的参数动态创建模型

    Args:
        model_name: 模型名称
        config: 配置对象，包含模型所需参数
        logger: 日志记录器
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {', '.join(AVAILABLE_MODELS.keys())}")

    model_class = AVAILABLE_MODELS[model_name]

    # 获取模型初始化函数的参数列表
    init_signature = inspect.signature(model_class.__init__)
    param_names = list(init_signature.parameters.keys())

    # 移除self参数
    if 'self' in param_names:
        param_names.remove('self')

    # 参数映射：模型参数名到配置属性名的映射
    param_mapping = {
        # 基本参数映射
        'input_channels': ['input_channels', 'in_channels', 'n_components'],
        'num_classes': ['num_classes', 'n_classes', 'classes'],
        'patch_size': ['patch_size', 'patch_sz', 'window_size'],
        # 可选参数映射
        'dropout': ['dropout', 'drop_rate'],
        'feature_dim': ['feature_dim', 'dim', 'hidden_dim'],
        'depth': ['depth', 'n_layers', 'num_layers'],
        'heads': ['heads', 'n_heads', 'num_heads'],
        'dim_head': ['dim_head', 'head_dim'],
        'd_state': ['d_state', 'state_dim'],
        'mlp_dim': ['mlp_dim', 'ffn_dim', 'ff_dim'],
        'expand': ['expand', 'expansion_factor', 'expansion'],
        # 添加更多可能的参数映射
    }

    # 构建模型参数字典
    model_params = {}

    # 遍历模型需要的每个参数
    for param in param_names:
        # 如果参数在映射中，尝试从配置中获取对应值
        if param in param_mapping:
            for config_attr in param_mapping[param]:
                if hasattr(config, config_attr):
                    model_params[param] = getattr(config, config_attr)
                    break
        # 直接检查配置中是否有同名属性
        elif hasattr(config, param):
            model_params[param] = getattr(config, param)

    # 确保基本参数存在
    if 'input_channels' in param_names and 'input_channels' not in model_params:
        raise ValueError(f"模型{model_name}需要input_channels参数，但在配置中未找到")

    if 'num_classes' in param_names and 'num_classes' not in model_params:
        raise ValueError(f"模型{model_name}需要num_classes参数，但在配置中未找到")

    # 创建模型实例
    model = model_class(**model_params)
    model.apply(init_weights)
    device = get_device()

    # 记录实际使用的参数
    if logger:
        logger.info("模型创建参数:")
        for param_name, param_value in model_params.items():
            logger.info(f"  - {param_name}: {param_value}")

    return model.to(device)
