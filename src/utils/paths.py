# src/utils/paths.py

import os
import platform
import re
from datetime import datetime


def get_project_root():
    """获取项目根目录"""
    # 假设此文件位于 src/utils/paths.py
    current_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_path)))


def sanitize_filename(filename):
    """清理文件名，移除不同操作系统的非法字符"""
    if platform.system() == 'Windows':
        illegal_chars = r'[\\/:*?"<>|]'  # Windows不允许的字符
    else:
        illegal_chars = r'[/]'  # Unix/Mac主要是斜杠

    return re.sub(illegal_chars, '_', filename)


def ensure_dir(directory):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)
    return directory


# 定义常用路径常量
ROOT_DIR = get_project_root()
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TEST_RESULTS_DIR = os.path.join(ROOT_DIR, "test_results")
COMPARE_RESULTS_DIR = os.path.join(ROOT_DIR, "compare_results")
ensure_dir(COMPARE_RESULTS_DIR)

# 确保基本目录存在
for directory in [RESULTS_DIR, TEST_RESULTS_DIR]:
    ensure_dir(directory)


def get_optuna_dir(save_dir, trial_number):
    """获取Optuna试验目录"""
    trial_dir = os.path.join(save_dir, f'optuna_trial_{trial_number}')
    return ensure_dir(trial_dir)


def get_plot_path(save_dir, filename="best_lr_schedule.png"):
    """获取图表保存路径"""
    return os.path.join(save_dir, sanitize_filename(filename))


def create_experiment_dir(dataset_name, model_name, is_main=True):
    """创建实验目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    folder_name = f"{dataset_name}_{model_name}_{timestamp}"

    if is_main:
        save_dir = os.path.join(RESULTS_DIR, folder_name)
    else:
        save_dir = os.path.join(TEST_RESULTS_DIR, folder_name)

    return ensure_dir(save_dir)


def create_comparison_dir(dataset_name=None, model_name=None):
    """创建比较结果目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')

    if dataset_name and model_name:
        folder_name = f"{dataset_name}_{model_name}_{timestamp}"
    else:
        folder_name = f"comparison_{timestamp}"

    save_dir = os.path.join(COMPARE_RESULTS_DIR, folder_name)
    return ensure_dir(save_dir)


def get_file_path(base_dir, filename):
    """获取安全的文件路径"""
    return os.path.join(base_dir, sanitize_filename(filename))
