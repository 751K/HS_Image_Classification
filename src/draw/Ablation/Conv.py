import matplotlib.pyplot as plt
import numpy as np

# 指标名称
metrics = ['OA', 'AA', 'Kappa']
x = np.arange(len(metrics))
width = 0.2  # 每个柱子的宽度

# 自定义颜色
color_baseline = "#005C53"
color_remove3d = "#9FC131"
color_remove2d = "#DBF227"
color_residual = "#D6D58E"

# Salinas 数据
salinas_baseline = [0.9671, 0.9467, 0.9562]
salinas_remove3d = [0.9378, 0.8807, 0.9170]
salinas_remove2d = [0.9159, 0.8662, 0.8879]
salinas_residual = [0.9629, 0.9295, 0.9506]

# Pavia 数据
pavia_baseline = [0.9698, 0.9534, 0.9599]
pavia_remove3d = [0.9115, 0.8452, 0.8808]
pavia_remove2d = [0.9188, 0.8602, 0.8914]
pavia_residual = [0.8612, 0.7530, 0.8122]

# Indian Pines 数据
indian_baseline = [0.9763, 0.8871, 0.9730]
indian_remove3d = [0.9611, 0.8028, 0.9556]
indian_remove2d = [0.9417, 0.7935, 0.9335]
indian_residual = [0.6800, 0.5324, 0.6307]

# 创建 1 行 3 列子图，每个子图对应一个数据集
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Salinas 子图
axs[0].bar(x - 1.5 * width, salinas_baseline, width, label='baseline', color=color_baseline)
axs[0].bar(x - 0.5 * width, salinas_remove3d, width, label='Remove 3D Conv', color=color_remove3d)
axs[0].bar(x + 0.5 * width, salinas_remove2d, width, label='Remove 2D Conv', color=color_remove2d)
axs[0].bar(x + 1.5 * width, salinas_residual, width, label='Add Residual', color=color_residual)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics)
axs[0].set_title('Salinas')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_ylim(0.50, 1.00)  # 压缩纵轴范围

# Pavia 子图
axs[1].bar(x - 1.5 * width, pavia_baseline, width, label='baseline', color=color_baseline)
axs[1].bar(x - 0.5 * width, pavia_remove3d, width, label='Remove 3D Conv', color=color_remove3d)
axs[1].bar(x + 0.5 * width, pavia_remove2d, width, label='Remove 2D Conv', color=color_remove2d)
axs[1].bar(x + 1.5 * width, pavia_residual, width, label='Add Residual', color=color_residual)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics)
axs[1].set_title('Pavia')
axs[1].legend()
axs[1].set_ylim(0.50, 1.00)

# Indian Pines 子图
axs[2].bar(x - 1.5 * width, indian_baseline, width, label='baseline', color=color_baseline)
axs[2].bar(x - 0.5 * width, indian_remove3d, width, label='Remove 3D Conv', color=color_remove3d)
axs[2].bar(x + 0.5 * width, indian_remove2d, width, label='Remove 2D Conv', color=color_remove2d)
axs[2].bar(x + 1.5 * width, indian_residual, width, label='Add Residual', color=color_residual)
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics)
axs[2].set_title('Indian Pines')
axs[2].legend()
axs[2].set_ylim(0.50, 1.00)

plt.suptitle('Convolution Ablation Experiment: Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
