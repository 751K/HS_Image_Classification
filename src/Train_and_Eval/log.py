import logging
import os
from datetime import datetime


def setup_logger(save_dir):
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