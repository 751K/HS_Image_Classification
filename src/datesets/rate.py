import numpy as np
import pandas as pd
import logging
import argparse
import os
import sys
from tabulate import tabulate
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datesets.datasets_load import load_dataset
from src.datesets.Dataset import prepare_data
from src.config import Config
from src.utils.paths import ensure_dir, ROOT_DIR, sanitize_filename


def analyze_class_distribution(dataset_name, test_size=None, random_state=42, dim=1, patch_size=9):
    """分析数据集中各类别在不同数据集中的分布情况"""
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 加载配置
    config = Config()

    # 加载数据集
    logger.info(f"加载{dataset_name}数据集...")
    data, labels, label_names = load_dataset(dataset_name, logger)
    config.update_dataset_config(dataset_name)
    if test_size is None:
        test_size = config.test_size
    # 准备数据
    logger.info(f"准备数据，维度: {dim}, 测试集比例: {test_size}")
    train_data, train_labels, test_data, test_labels, val_data, val_labels = prepare_data(
        data, labels, test_size=test_size, random_state=random_state, dim=dim, patch_size=patch_size, logger=logger
    )

    # 计算样本数量
    class_counts = {
        'train': np.bincount(train_labels + 1, minlength=len(label_names))[1:],
        'val': np.bincount(val_labels + 1, minlength=len(label_names))[1:],
        'test': np.bincount(test_labels + 1, minlength=len(label_names))[1:]
    }

    # 构建结果表格
    result_data = []
    class_names = label_names[1:]  # 排除背景类

    for i, class_name in enumerate(class_names):
        class_id = i + 1
        train_count = class_counts['train'][i]
        val_count = class_counts['val'][i]
        test_count = class_counts['test'][i]
        total_samples = train_count + val_count + test_count

        if total_samples > 0:
            row = {
                '类别ID': class_id,
                '类别名称': class_name,
                '训练集': train_count,
                '验证集': val_count,
                '测试集': test_count,
                '总数': total_samples,
                '训练集%': (train_count / total_samples * 100) if total_samples > 0 else 0,
                '验证集%': (val_count / total_samples * 100) if total_samples > 0 else 0,
                '测试集%': (test_count / total_samples * 100) if total_samples > 0 else 0
            }
            result_data.append(row)

    # 添加总计行
    total_train = sum(class_counts['train'])
    total_val = sum(class_counts['val'])
    total_test = sum(class_counts['test'])
    total_all = total_train + total_val + total_test

    total_row = {
        '类别ID': '总计',
        '类别名称': '',
        '训练集': total_train,
        '验证集': total_val,
        '测试集': total_test,
        '总数': total_all,
        '训练集%': (total_train / total_all * 100) if total_all > 0 else 0,
        '验证集%': (total_val / total_all * 100) if total_all > 0 else 0,
        '测试集%': (total_test / total_all * 100) if total_all > 0 else 0
    }
    result_data.append(total_row)

    df = pd.DataFrame(result_data)
    return df, {'dim': dim, 'test_size': test_size, 'patch_size': patch_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析高光谱数据集各类别样本分布情况')
    parser.add_argument('--dataset', type=str, default='Wuhan',
                        help='数据集名称 ')
    parser.add_argument('--test_size', type=float, default=None,
                        help='测试集比例')
    parser.add_argument('--dim', type=int, default=2,
                        help='数据维度 ')
    parser.add_argument('--patch_size', type=int, default=7,
                        help='patch大小 ')
    parser.add_argument('--random_state', type=int, default=3407,
                        help='随机种子 ')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径')

    args = parser.parse_args()

    # 分析类别分布
    df, info = analyze_class_distribution(
        args.dataset,
        test_size=args.test_size,
        dim=args.dim,
        patch_size=args.patch_size,
        random_state=args.random_state
    )

    # 打印结果
    print(f"\n{args.dataset}数据集类别分布分析:")
    print(f"维度: {info['dim']}, 测试集比例: {info['test_size']}, Patch尺寸: {info['patch_size']}")

    # 格式化表格输出
    table = tabulate(df, headers='keys', tablefmt='pretty', floatfmt=".2f")
    print(table)

    # 保存结果到CSV文件
    if args.output:
        # 用户指定了输出路径
        output_path = args.output
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    else:
        # 自动创建输出目录和文件名
        data_analysis_dir = os.path.join(ROOT_DIR, "data_analysis")
        ensure_dir(data_analysis_dir)

        # 创建特定于此数据集的子目录
        dataset_dir = os.path.join(data_analysis_dir, args.dataset)
        ensure_dir(dataset_dir)

        # 生成文件名，包含关键参数
        timestamp = datetime.now().strftime('%m%d_%H%M')
        filename = f"{args.dataset}_dim{args.dim}_patch{args.patch_size}_test{info['test_size']}_{timestamp}.csv"
        output_path = os.path.join(dataset_dir, sanitize_filename(filename))

    # 保存CSV文件
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
