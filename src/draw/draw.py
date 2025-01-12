import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_roc_curve(y_true, y_scores, n_classes):
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve of class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_scores, n_classes):
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
        avg_precision = average_precision_score(y_true == i, y_scores[:, i])
        plt.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()


def plot_class_distribution(y_true, y_pred, n_classes):
    class_counts = [sum(y_true == i) for i in range(n_classes)]
    correct_counts = [sum((y_true == i) & (y_pred == i)) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(n_classes)
    width = 0.35

    ax.bar(x, class_counts, width, label='Total Samples')
    ax.bar([i + width for i in x], correct_counts, width, label='Correctly Classified')

    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution and Classification Results')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
    ax.legend()

    plt.show()


def plot_misclassified_samples(X_test, y_true, y_pred, n_samples=5):
    misclassified = np.where(y_true != y_pred)[0]
    n_samples = min(n_samples, len(misclassified))

    fig, axes = plt.subplots(1, n_samples, figsize=(20, 4))
    for i, idx in enumerate(np.random.choice(misclassified, n_samples, replace=False)):
        axes[i].imshow(X_test[idx].transpose(1, 2, 0))  # Assuming CHW format
        axes[i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
