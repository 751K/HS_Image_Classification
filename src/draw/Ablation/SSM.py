import matplotlib.pyplot as plt
import numpy as np

# 指标名称
metrics = ['OA', 'AA', 'Kappa']
x = np.arange(len(metrics))
width = 0.25  # 每个柱子的宽度

# Salinas 数据
salinas_baseline = [0.9698, 0.9534, 0.9599]
salinas_ssm = [0.9510, 0.9259, 0.9347]

# Pavia 数据
pavia_baseline = [0.9698, 0.9534, 0.9599]
pavia_ssm = [0.9510, 0.9259, 0.9347]

# Indian Pines 数据
indian_baseline = [0.9763, 0.8871, 0.9730]
indian_ssm = [0.9713, 0.8931, 0.9673]

# 自定义颜色
color_baseline = "#005C53"
color_ssm = "#9FC131"

# 创建 1 行 3 列子图，每个子图对应一个数据集
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Salinas 子图
axs[0].bar(x - width / 2, salinas_baseline, width, label='baseline', color=color_baseline)
axs[0].bar(x + width / 2, salinas_ssm, width, label='SSM', color=color_ssm)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics)
axs[0].set_title('Salinas')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_ylim(0.85, 1.00)

# Pavia 子图
axs[1].bar(x - width / 2, pavia_baseline, width, label='baseline', color=color_baseline)
axs[1].bar(x + width / 2, pavia_ssm, width, label='SSM', color=color_ssm)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics)
axs[1].set_title('Pavia')
axs[1].legend()
axs[1].set_ylim(0.85, 1.00)

# Indian Pines 子图
axs[2].bar(x - width / 2, indian_baseline, width, label='baseline', color=color_baseline)
axs[2].bar(x + width / 2, indian_ssm, width, label='SSM', color=color_ssm)
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics)
axs[2].set_title('Indian Pines')
axs[2].legend()
axs[2].set_ylim(0.85, 1.00)

plt.suptitle('Mamba Fusion: Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
