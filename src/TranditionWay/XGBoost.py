import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple
from tqdm import tqdm

from config import Config
from src.datesets.Dataset import prepare_data
from src.datesets.datasets_load import load_dataset


class XGBoostHSI:
    """
    使用XGBoost进行高光谱图像分类的模型。
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1,
                 random_state: int = 42):
        """
        初始化XGBoost分类器。

        Args:
            n_estimators: 弱学习器数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            random_state: 随机种子
        """
        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,  # 使用所有可用CPU
            use_label_encoder=False,
            eval_metric='mlogloss'  # 多分类评估指标
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练XGBoost模型"""
        X_flat = X.reshape(X.shape[0], -1)
        print("训练XGBoost模型...")
        with tqdm(total=100, desc="训练进度") as pbar:
            self.classifier.fit(X_flat, y, verbose=False)
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
    # 加载数据集
    config = Config()
    data, labels, dataset_info = load_dataset(config.datasets)

    # 准备数据
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
        data, labels, test_size=config.test_size, random_state=config.seed,
        dim=3, patch_size=config.patch_size,
    )

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}, 验证集大小: {X_val.shape[0]}")

    # 创建和训练XGBoost模型
    xgb_model = XGBoostHSI(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    # 评估模型
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    # 训练集评估
    train_accuracy, train_report = xgb_model.evaluate(X_train, y_train)
    y_train_pred = xgb_model.predict(X_train)
    train_kappa = cohen_kappa_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_aa = np.mean(np.diag(train_cm) / np.sum(train_cm, axis=1))

    # 验证集评估
    val_accuracy, val_report = xgb_model.evaluate(X_val, y_val)
    y_val_pred = xgb_model.predict(X_val)
    val_kappa = cohen_kappa_score(y_val, y_val_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    val_aa = np.mean(np.diag(val_cm) / np.sum(val_cm, axis=1))

    # 测试集评估
    test_accuracy, test_report = xgb_model.evaluate(X_test, y_test)
    y_test_pred = xgb_model.predict(X_test)
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