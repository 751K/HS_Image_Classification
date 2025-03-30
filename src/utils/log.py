import logging
import os
from datetime import datetime


def setup_logger(save_dir: str) -> logging.Logger:
    """
    设置并配置一个日志记录器。

    Args:
        save_dir (str): 保存日志文件的目录路径。

    Returns:
        logging.Logger: 配置好的日志记录器实例。

    """
    # 创建一个日志记录器
    logger = logging.getLogger('HSI_Classification')
    logger.setLevel(logging.INFO)

    # 创建一个文件处理器，将日志写入文件
    log_file = os.path.join(save_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个格式器，定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_training_details(logger: logging.Logger, config) -> None:
    """
    记录配置参数和设备信息。

    Args:
        logger (logging.Logger): 日志记录器实例。
        config: 配置对象，包含模型名称、设备信息等。

    """
    logger.info(
        f"配置参数：epochs={config.num_epochs}, batch_size={config.batch_size}, num_workers={config.num_workers}")
    logger.info(f"预热比例: {config.warmup_ratio}, 学习率: {config.learning_rate}")
    logger.info(f"测试集比例: {config.test_size}")
    logger.info(f"随机种子: {config.seed}, 数据集: {config.datasets}, 衍生块大小: {config.patch_size}")
    logger.info(f'恢复检查点: {config.resume_checkpoint}')
    if config.stop_train:
        logger.info(f"早停耐心值: {config.patience}, 最小改进: {config.min_delta}")
    logger.info(f'保存目录: {config.save_dir}')
    if config.perform_dim_reduction:
        logger.info(f'降维方法: {config.dim_reduction}, 降维组件数: {config.n_components}')
