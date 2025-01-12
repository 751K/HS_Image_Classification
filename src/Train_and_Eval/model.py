# model.py
import json
import random

import numpy as np
import torch


def set_seed(seed):
    """
    设置随机种子以确保结果可复现。
    Args:
        seed (int): 用于随机数生成器的种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(state_dict, path):
    """
    保存模型的状态字典。

    Args:
        state_dict (dict): 模型的状态字典。
        path (str): 保存路径。
    """
    torch.save(state_dict, path)
    print(f"Model state_dict saved to {path}")


def save_test_results(all_preds, all_labels, accuracy, classification_report, path):
    """
    保存测试结果。

    Args:
        all_preds (list): 所有预测标签。
        all_labels (list): 所有真实标签。
        accuracy (float): 测试准确率。
        classification_report (str): 分类报告。
        path (str): 保存路径。
    """

    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    results = {
        "predictions": all_preds,
        "true_labels": all_labels,
        "accuracy": float(accuracy),
        "classification_report": classification_report
    }

    with open(path, 'w') as f:
        json.dump(results, f, indent=4, default=convert)

    print(f"Test results saved to {path}")
