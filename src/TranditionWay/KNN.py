import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple
from tqdm import tqdm

from config import Config
from src.datesets.Dataset import prepare_data
from src.datesets.datasets_load import load_dataset


class KNNHSI:
    """
    使用K近邻算法进行高光谱图像分类的模型。
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', algorithm: str = 'auto'):
        """
        初始化KNN分类器。

        Args:
            n_neighbors: 考虑的邻居数量
            weights: 权重类型，'uniform'为均匀权重，'distance'为距离加权
            algorithm: KNN算法类型，可选'auto', 'ball_tree', 'kd_tree', 'brute'
        """
        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            n_jobs=-1  # 并行计算
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练KNN模型"""
        X_flat = X.reshape(X.shape[0], -1)
        print("训练KNN模型...")
        with tqdm(total=100, desc="训练进度") as pbar:
            self.classifier.fit(X_flat, y)
            pbar.update(100)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测"""
        X_flat = X.reshape(X.shape[0], -1)
        return self.classifier.predict(X_flat)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """评估模型性能"""
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report


if __name__ == '__main__':
    config = Config()
    data, labels, dataset_info = load_dataset(config.datasets)

    # 准备数据
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
        data, labels, test_size=config.test_size, random_state=config.seed,
        dim=3, patch_size=config.patch_size,
    )

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}, 验证集大小: {X_val.shape[0]}")

    # 创建和训练KNN模型
    knn_model = KNNHSI(n_neighbors=5, weights='distance', algorithm='auto')
    knn_model.fit(X_train, y_train)

    # 评估模型
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    # 训练集评估
    train_accuracy, train_report = knn_model.evaluate(X_train, y_train)
    y_train_pred = knn_model.predict(X_train)
    train_kappa = cohen_kappa_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_aa = np.mean(np.diag(train_cm) / np.sum(train_cm, axis=1))

    # 验证集评估
    val_accuracy, val_report = knn_model.evaluate(X_val, y_val)
    y_val_pred = knn_model.predict(X_val)
    val_kappa = cohen_kappa_score(y_val, y_val_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    val_aa = np.mean(np.diag(val_cm) / np.sum(val_cm, axis=1))

    # 测试集评估
    test_accuracy, test_report = knn_model.evaluate(X_test, y_test)
    y_test_pred = knn_model.predict(X_test)
    test_kappa = cohen_kappa_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_aa = np.mean(np.diag(test_cm) / np.sum(test_cm, axis=1))

    # 打印评估结果
    print(f"训练集结果 - OA: {train_accuracy:.4f}, AA: {train_aa:.4f}, Kappa: {train_kappa:.4f}")
    print("训练集分类报告:")
    print(train_report)

    print(f"验证集结果 - OA: {val_accuracy:.4f}, AA: {val_aa:.4f}, Kappa: {val_kappa:.4f}")
    print("验证集分类报告:")
    print(val_report)

    print(f"测试集结果 - OA: {test_accuracy:.4f}, AA: {test_aa:.4f}, Kappa: {test_kappa:.4f}")
    print("测试集分类报告:")
    print(test_report)