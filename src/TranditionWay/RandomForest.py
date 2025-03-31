import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple

from config import Config
from src.datesets.Dataset import prepare_data
from src.datesets.datasets_load import load_dataset


class RandomForestHSI:
    """
    使用随机森林进行高光谱图像分类的模型。
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        """
        初始化随机森林分类器。

        Args:
            n_estimators (int): 森林中树的数量。默认为100。
            max_depth (int): 树的最大深度。默认为None（无限制）。
            random_state (int): 随机数生成器的种子。默认为42。
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # 使用所有可用的CPU核心
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练随机森林模型。

        Args:
            X (np.ndarray): 训练数据。
            y (np.ndarray): 训练标签。
        """
        # 将输入数据展平为2D
        X_flat = X.reshape(X.shape[0], -1)
        self.classifier.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测。

        Args:
            X (np.ndarray): 输入数据。

        Returns:
            np.ndarray: 预测的类别。
        """
        X_flat = X.reshape(X.shape[0], -1)
        return self.classifier.predict(X_flat)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """
        评估模型性能。

        Args:
            X (np.ndarray): 测试数据。
            y (np.ndarray): 真实标签。

        Returns:
            Tuple[float, str]: 包含准确率和详细分类报告的元组。
        """
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

    # 创建和训练模型
    rf_model = RandomForestHSI(n_estimators=100, max_depth=10)
    rf_model.fit(X_train, y_train)

    # 评估模型
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    # 训练集评估
    train_accuracy, train_report = rf_model.evaluate(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    train_kappa = cohen_kappa_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_aa = np.mean(np.diag(train_cm) / np.sum(train_cm, axis=1))

    # 验证集评估
    val_accuracy, val_report = rf_model.evaluate(X_val, y_val)
    y_val_pred = rf_model.predict(X_val)
    val_kappa = cohen_kappa_score(y_val, y_val_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    val_aa = np.mean(np.diag(val_cm) / np.sum(val_cm, axis=1))

    # 测试集评估
    test_accuracy, test_report = rf_model.evaluate(X_test, y_test)
    y_test_pred = rf_model.predict(X_test)
    test_kappa = cohen_kappa_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_aa = np.mean(np.diag(test_cm) / np.sum(test_cm, axis=1))

    print(f"训练集结果 - OA: {train_accuracy:.4f}, AA: {train_aa:.4f}, Kappa: {train_kappa:.4f}")
    print("训练集分类报告:")
    print(train_report)

    print(f"验证集结果 - OA: {val_accuracy:.4f}, AA: {val_aa:.4f}, Kappa: {val_kappa:.4f}")
    print("验证集分类报告:")
    print(val_report)

    print(f"测试集结果 - OA: {test_accuracy:.4f}, AA: {test_aa:.4f}, Kappa: {test_kappa:.4f}")
    print("测试集分类报告:")
    print(test_report)
