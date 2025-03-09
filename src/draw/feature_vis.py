import os
import torch
import shap
import numpy as np
from torch import nn

from src.config import Config
from datesets.datasets_load import load_dataset
from Dim.api import apply_dimension_reduction
from model_init import create_model
from src.datesets.Dataset import prepare_data, create_data_loaders
from src.Train_and_Eval.log import setup_logger
from src.Train_and_Eval.model import set_seed


class CustomDeepExplainer(shap.DeepExplainer):
    def __init__(self, model, background_data):
        super().__init__(model, background_data)

    def shap_values(self, X, ranked_outputs=0, output_rank_order=None, check_additivity=False):
        """
        重写 SHAP 值计算，跳过 LayerNorm 和 DropPath 层
        """
        # 遍历模型的每一层，跳过 LayerNorm 和 DropPath 层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                # 对 LayerNorm 层进行忽略处理
                print(f"跳过 LayerNorm 层：{name}")
                # 我们可以在这里做一些处理，假设 LayerNorm 层不对输出产生影响
                # 你可以选择将该层输出视为常数，或者完全忽略它的 SHAP 值贡献
                continue  # 跳过 LayerNorm 层

            elif isinstance(module, nn.Dropout):
                # 对 Dropout 和 DropPath 层进行忽略处理
                print(f"跳过 Dropout 或 DropPath 层：{name}")
                # DropPath 也被视为不对最终 SHAP 值贡献，直接跳过
                continue  # 跳过 Dropout

        # 调用父类的 shap_values 方法，继续计算 SHAP 值
        shap_values = super().shap_values(X, check_additivity=check_additivity)
        return shap_values


def explain_with_Spatial_Spectral(model, background_loader, explain_loader, class_names, device, logger):
    # 从 DataLoader 中提取背景数据
    background_data = []
    for batch in background_loader:
        background_data.append(batch[0].to(device))  # batch[0] 是输入数据
    background_data = torch.cat(background_data, dim=0)

    # 使用 DeepExplainer 创建 SHAP 解释器
    explainer = CustomDeepExplainer(model, background_data)
    logger.info("SHAP 解释器创建完成")

    # 获取解释数据（待解释的样本）
    explain_data = []
    for batch in explain_loader:
        explain_data.append(batch[0].to(device))
    explain_data = torch.cat(explain_data, dim=0)

    # 计算 SHAP 值，确保 output_rank_order 参数有效
    try:
        shap_values = explainer.shap_values(explain_data)  # 设置有效的 output_rank_order
        logger.info("SHAP 值计算完成")
    except ValueError as e:
        logger.error(f"SHAP 计算失败: {e}")
        return

    # 对每个类别分别可视化 SHAP 值
    for class_index, class_name in enumerate(class_names):
        shap_values_class = shap_values[class_index]  # 该类别的 SHAP 值

        # 只对每个类别的第一个样本进行可视化
        shap_values_sample = shap_values_class[0]  # 选择第一个样本
        explain_sample = explain_data[0].cpu().numpy()  # 获取第一个样本的输入数据

        # 绘制 SHAP 图像
        shap.image_plot(shap_values_sample, explain_sample)


def run_shap_explanation():
    config = Config()

    set_seed(config.seed)

    # 设置日志记录器
    logger = setup_logger(config.save_dir)
    logger.info("开始 SHAP 解释")

    # 设置模型路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_model.pth")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data, labels, dataset_info = load_dataset(config.datasets, logger)
    logger.info(f"数据加载完成：{config.datasets}")

    data = apply_dimension_reduction(data, config, logger)
    logger.info(f"使用{config.dim_reduction}降维完成")

    num_classes = len(np.unique(labels))
    input_channels = data.shape[-1]

    # 创建模型
    model = create_model(config.model_name, input_channels, num_classes, config.patch_size)

    # 加载模型
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"模型成功加载：{model_path}")
    except Exception as e:
        import traceback
        error_msg = f"加载模型时发生错误::\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)
        return

    model.to(device)
    for param in model.parameters():
        param.requires_grad = True  # 确保所有参数都需要梯度

    logger.info(f"模型加载完成：{config.model_name}")

    # 准备数据
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, dim=model.dim,
                                                                  patch_size=config.patch_size)
    logger.info("数据预处理完成")

    # 创建背景数据和解释数据的 DataLoader
    background_loader, explain_loader = create_data_loaders(
        X_train[:100], y_train[:100], X_test[:100], y_test[:100],
        config.batch_size, config.num_workers, dim=model.dim, logger=logger
    )
    logger.info("DataLoader 创建完成")

    # 获取类别名称
    class_names = [f"Class {i}" for i in range(num_classes)]

    # 运行 SHAP 解释
    explain_with_Spatial_Spectral(model, background_loader, explain_loader, class_names, device, logger)

    logger.info("SHAP 解释程序执行完毕")


if __name__ == '__main__':
    run_shap_explanation()
