# model.py
import json
import random

import numpy as np
import torch
from sympy.abc import kappa


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


def save_test_results( accuracy, path, logger):
    """
    保存测试结果。

    Args:
        accuracy (float): 测试准确率。
        path (str): 保存路径。
        logger: logger
    """

    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    oa, aa, kappa = accuracy
    results = {
        "overall_accuracy": float(oa),
        "average_accuracy": float(aa),
        "kappa": float(kappa),
    }

    with open(path, 'w') as f:
        json.dump(results, f, indent=4, default=convert)

    logger.info(f"Test results saved to {path}")
