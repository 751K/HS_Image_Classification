import os
import numpy as np
from scipy.io import loadmat


def load_dataset(dataset_name,logger=None):
    """
    加载高光谱数据集（Indian Pines, Pavia University, Salinas）

    Args:
    dataset_name (str): 数据集名称 ('Indian', 'Pavia', 'Salinas')
    base_path (str): 项目根目录路径

    Return:
    tuple: (data, labels, label_names)
        - data (numpy.ndarray): 形状为 (H, W, C) 的高光谱图像数据
        - labels (numpy.ndarray): 形状为 (H, W) 的标签图像
        - label_names (list): 标签名称列表
    """
    dataset_info = {
        'Indian': {
            'data_file': 'Indian_pines_corrected.mat',
            'label_file': 'Indian_pines_gt.mat',
            'data_key': 'indian_pines_corrected',
            'label_key': 'indian_pines_gt',
            'label_names': ['Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                            'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                            'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                            'Stone-Steel-Towers']
        },
        'Pavia': {
            'data_file': 'PaviaU.mat',
            'label_file': 'PaviaU_gt.mat',
            'data_key': 'paviaU',
            'label_key': 'paviaU_gt',
            'label_names': ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                            'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
        },
        'Salinas': {
            'data_file': 'Salinas_corrected.mat',
            'label_file': 'Salinas_gt.mat',
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt',
            'label_names': ['Background', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow',
                            'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained',
                            'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                            'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                            'Vinyard_untrained', 'Vinyard_vertical_trellis']
        },
        'KSC': {
            'data_file': 'KSC.mat',
            'label_file': 'KSC_gt.mat',
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'label_names': ['Undefined', 'Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak', 'Slash pine',
                            'Oak/Broadleaf', 'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
                            'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water']
        },
        'Botswana': {
            'data_file': 'Botswana.mat',
            'label_file': 'Botswana_gt.mat',
            'data_key': 'Botswana',
            'label_key': 'Botswana_gt',
            'label_names': ['Undefined', 'Water', 'Hippo grass', 'Floodplain grasses 1', 'Floodplain grasses 2',
                            'Reeds', 'Riparian', 'Firescar', 'Island interior', 'Acacia woodlands',
                            'Acacia shrublands', 'Acacia grasslands', 'Short mopane', 'Mixed mopane',
                            'Exposed soils']
        }
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(os.path.dirname(current_dir))
    if logger:
        logger.info(f"项目根目录: {base_path}")
    else:
        print("项目根目录:", base_path)

    if dataset_name not in dataset_info:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets are: {', '.join(dataset_info.keys())}")

    info = dataset_info[dataset_name]

    # 构建数据文件的完整路径
    data_path = os.path.join(base_path, 'datasets', dataset_name)
    data_file = os.path.join(data_path, info['data_file'])
    label_file = os.path.join(data_path, info['label_file'])

    # 打印文件路径以进行调试
    if logger:
        logger.info(f"尝试加载数据文件: {data_file}")
        logger.info(f"尝试加载标签文件: {label_file}")
    else:
        print(f"尝试加载数据文件: {data_file}")
        print(f"尝试加载标签文件: {label_file}")

    # 加载数据
    data = loadmat(data_file)[info['data_key']]
    labels = loadmat(label_file)[info['label_key']]

    # 确保数据类型正确
    data = data.astype(np.float32)
    labels = labels.astype(np.int64)

    return data, labels, info['label_names']


# 使用示例
if __name__ == "__main__":
    dataset_names = ['Indian', 'Pavia', 'Salinas', 'KSC', 'Botswana']

    for dataset_name in dataset_names:
        print(f"\n加载 {dataset_name} 数据集:")
        try:
            data, labels, label_names = load_dataset(dataset_name)
            print("数据形状:", data.shape)
            print("标签形状:", labels.shape)
            print("类别数量:", len(np.unique(labels)) - 1)  # 减去背景类
            print("标签名称:", label_names)
            print("数据范围:", np.min(data), "-", np.max(data))
            print("标签值:", np.unique(labels))
        except Exception as e:
            print(f"加载 {dataset_name} 数据集时出错: {str(e)}")
