from src.datesets.Dataset import prepare_data
from src.datesets.datasets_load import load_dataset

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import os


class SVMHSI:
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', random_state: int = 42):
        self.classifier = SVC(C=C, kernel=kernel, random_state=random_state, verbose=True)
        self.patch_size = None  # 将在 fit 方法中设置

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        if self.patch_size is None:
            return X.reshape(X.shape[0], -1)
        else:
            # 假设 X 的形状是 (samples, channels, height, width)
            samples, channels, height, width = X.shape
            patches = np.lib.stride_tricks.sliding_window_view(
                X, (channels, self.patch_size, self.patch_size)
            ).reshape(-1, channels * self.patch_size * self.patch_size)
            return patches

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X.shape) == 4:  # (samples, channels, height, width)
            self.patch_size = X.shape[2]  # 假设高度和宽度相等
        X_preprocessed = self.preprocess_data(X)
        print("Training SVM model...")
        with tqdm(total=100, desc="Training Progress", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            self.classifier.fit(X_preprocessed, y)
            pbar.update(100)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_preprocessed = self.preprocess_data(X)
        return self.classifier.predict(X_preprocessed)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        print("Evaluating model...")
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report

    def visualize_classification(self, data: np.ndarray, labels: np.ndarray, class_names: list, save_path: str = None):
        print("Generating classification visualization...")
        height, width, channels = data.shape

        # 创建滑动窗口
        if self.patch_size:
            padded_data = np.pad(data, ((self.patch_size // 2, self.patch_size // 2),
                                        (self.patch_size // 2, self.patch_size // 2),
                                        (0, 0)), mode='reflect')
            windows = np.lib.stride_tricks.sliding_window_view(
                padded_data, (self.patch_size, self.patch_size, channels)
            )
            windows = windows.reshape(-1, channels, self.patch_size, self.patch_size)
        else:
            windows = data.reshape(-1, channels)

        # 逐批次进行预测
        batch_size = 1000  # 可以根据内存情况调整
        predictions = []
        for i in tqdm(range(0, windows.shape[0], batch_size), desc="Classifying pixels"):
            batch = windows[i:i + batch_size]
            predictions.extend(self.predict(batch))

        # 重塑预测结果
        classification_map = np.array(predictions).reshape(height, width)

        # 创建颜色映射
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.jet(np.linspace(0, 1, num_classes))

        # 创建分类结果的彩色图
        rgb_image = colors[classification_map]

        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 原始标签
        ax1.imshow(labels, cmap='jet')
        ax1.set_title('Original Labels')
        ax1.axis('off')

        # 分类结果
        ax2.imshow(classification_map, cmap='jet')
        ax2.set_title('Classification Result')
        ax2.axis('off')

        # 创建图例
        legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=class_names[i])
                           for i in range(len(class_names)) if i in unique_labels]

        # 在图像之间添加图例
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        if save_path is None:
            # 获取项目根目录并保存图像
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pic_dir = os.path.join(root_dir, "Pic")
            os.makedirs(pic_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            filename = f"svm_classification_visualization_{timestamp}.png"
            save_path = os.path.join(pic_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # 计算准确率
        mask = labels != 0  # 假设 0 是背景类
        accuracy = np.mean(classification_map[mask] == labels[mask])
        print(f"Classification accuracy: {accuracy:.4f}")

        return classification_map


if __name__ == '__main__':
    # 加载数据集
    print("Loading dataset...")
    data, labels, dataset_info = load_dataset('Pavia')

    # 准备数据
    print("Preparing data...")
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
        data, labels, test_size=0.95, random_state=42,
        dim=3, patch_size=5
    )

    # 创建和训练模型
    svm_model = SVMHSI(C=1.0, kernel='rbf')
    svm_model.fit(X_train, y_train)

    # 评估模型
    print("Evaluating on training set...")
    train_accuracy, train_report = svm_model.evaluate(X_train, y_train)
    print("Evaluating on validation set...")
    print("Evaluating on test set...")
    test_accuracy, test_report = svm_model.evaluate(X_test, y_test)

    print(f"\nTraining Accuracy: {train_accuracy}")
    print("Training Classification Report:")
    print(train_report)

    print(f"\nTest Accuracy: {test_accuracy}")
    print("Test Classification Report:")
    print(test_report)

    # 可视化分类结果
    svm_model.visualize_classification(data, labels, dataset_info)
