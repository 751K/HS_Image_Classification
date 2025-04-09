# src/utils/paths.py

import os
import platform
import re
from datetime import datetime


# 基础路径工具函数
def get_project_root():
    """获取项目根目录"""
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


def get_file_path(base_dir, filename):
    """获取安全的文件路径"""
    return os.path.join(base_dir, sanitize_filename(filename))


# 定义项目基本目录
ROOT_DIR = get_project_root()

# ================ 实验结果目录 ================
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TEST_RESULTS_DIR = os.path.join(ROOT_DIR, "test_results")
ensure_dir(RESULTS_DIR)
ensure_dir(TEST_RESULTS_DIR)


def create_experiment_dir(dataset_name, model_name, is_main=True):
    """创建实验目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    folder_name = f"{dataset_name}_{model_name}_{timestamp}"
    save_dir = os.path.join(RESULTS_DIR if is_main else TEST_RESULTS_DIR, folder_name)
    return ensure_dir(save_dir)


def get_optuna_dir(save_dir, trial_number):
    """获取Optuna试验目录"""
    trial_dir = os.path.join(save_dir, f'optuna_trial_{trial_number}')
    return ensure_dir(trial_dir)


def get_plot_path(save_dir, filename="best_lr_schedule.png"):
    """获取图表保存路径"""
    return os.path.join(save_dir, sanitize_filename(filename))


# ================ 比较结果目录 ================
COMPARE_RESULTS_DIR = os.path.join(ROOT_DIR, "compare_results")
ensure_dir(COMPARE_RESULTS_DIR)


def create_comparison_dir(dataset_name=None, model_name=None):
    """创建比较结果目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    if dataset_name and model_name:
        folder_name = f"{dataset_name}_{model_name}_{timestamp}"
    else:
        folder_name = f"comparison_{timestamp}"
    save_dir = os.path.join(COMPARE_RESULTS_DIR, folder_name)
    return ensure_dir(save_dir)


# ================ 批处理结果目录 ================
BATCH_RESULTS_DIR = os.path.join(ROOT_DIR, "batch_result")
ensure_dir(BATCH_RESULTS_DIR)


def create_batch_result_dir(dataset_name=None, patch_size=None):
    """创建批处理结果目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    if dataset_name and patch_size:
        folder_name = f"comparison_{dataset_name}_{patch_size}_{timestamp}"
    else:
        folder_name = f"comparison_{timestamp}"
    save_dir = os.path.join(BATCH_RESULTS_DIR, folder_name)
    return ensure_dir(save_dir)


# ================ 参数比较结果目录 ================
PARAM_COMPARISON_DIR = os.path.join(ROOT_DIR, "param_comparison")
ensure_dir(PARAM_COMPARISON_DIR)


def create_param_comparison_dir(model_name=None, parameters_name=None, dataset_name=None):
    """创建参数比较结果目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    if model_name and parameters_name and dataset_name:
        folder_name = f"{model_name}_{parameters_name}_{dataset_name}_{timestamp}"
    else:
        folder_name = f"param_comparison_{timestamp}"
    save_dir = os.path.join(PARAM_COMPARISON_DIR, folder_name)
    return ensure_dir(save_dir)


# ================ 模型分析结果目录 ================
MODEL_ANALYSIS_DIR = os.path.join(ROOT_DIR, "model_analysis")
ensure_dir(MODEL_ANALYSIS_DIR)


def create_model_analysis_dir(model_name=None, channels=None, patch_size=None):
    """创建模型分析结果目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    if model_name and channels and patch_size:
        folder_name = f"{model_name}_c{channels}_p{patch_size}_{timestamp}"
    else:
        folder_name = f"model_analysis_{timestamp}"
    save_dir = os.path.join(MODEL_ANALYSIS_DIR, folder_name)
    return ensure_dir(save_dir)
