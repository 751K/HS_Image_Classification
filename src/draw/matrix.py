from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_and_save_confusion_matrix(labels, preds, num_classes, save_path):
    """
    计算并保存混淆矩阵。

    Args:
        labels (list or np.ndarray): 真实标签。
        preds (list or np.ndarray): 预测标签。
        num_classes (int): 类别数量。
        save_path (str): 保存图像的路径。
    """
    # 计算混淆矩阵
    num_classes = num_classes - 1
    cm = confusion_matrix(labels, preds)

    # 可视化混淆矩阵
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # 保存图像
    plt.savefig(save_path)

    # 显示图像
    plt.show()
