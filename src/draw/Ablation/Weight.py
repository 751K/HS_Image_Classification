import matplotlib.pyplot as plt
import numpy as np

# 指标名称
metrics = ['OA', 'AA', 'Kappa']
x = np.arange(len(metrics))
width = 0.2  # 每个柱子的宽度

# 自定义颜色
color_baseline = "#005C53"
color_mean = "#9FC131"
color_triangle = "#DBF227"
color_new = "#D6D58E"

# Salinas 数据
salinas_baseline = [0.9698, 0.9534, 0.9599]
salinas_mean = [0.9671, 0.9467, 0.9562]
salinas_triangle = [0.9579, 0.9285, 0.9439]
salinas_new = [0.9688, 0.9923, 0.9841]

# Pavia 数据
pavia_baseline = [0.9698, 0.9534, 0.9599]
pavia_mean = [0.9671, 0.9467, 0.9562]
pavia_triangle = [0.9579, 0.9285, 0.9439]
pavia_new = [0.9677, 0.9493, 0.9571]

# Indian Pines 数据
indian_baseline = [0.9763, 0.8871, 0.9730]
indian_mean = [0.9782, 0.8904, 0.9751]
indian_triangle = [0.9689, 0.8750, 0.9645]
indian_new = [0.9795, 0.9264, 0.9766]

# 创建 1 行 3 列子图，每个子图对应一个数据集
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Salinas 子图
axs[0].bar(x - 1.5 * width, salinas_baseline, width, label='cos', color=color_baseline)
axs[0].bar(x - 0.5 * width, salinas_mean, width, label='mean', color=color_mean)
axs[0].bar(x + 0.5 * width, salinas_triangle, width, label='triangle', color=color_triangle)
axs[0].bar(x + 1.5 * width, salinas_new, width, label='Ours', color=color_new)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics)
axs[0].set_title('Salinas')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_ylim(0.86, 1.00)  # 压缩纵轴范围

# Pavia 子图
axs[1].bar(x - 1.5 * width, pavia_baseline, width, label='cos', color=color_baseline)
axs[1].bar(x - 0.5 * width, pavia_mean, width, label='mean', color=color_mean)
axs[1].bar(x + 0.5 * width, pavia_triangle, width, label='triangle', color=color_triangle)
axs[1].bar(x + 1.5 * width, pavia_new, width, label='Ours', color=color_new)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics)
axs[1].set_title('Pavia')
axs[1].legend()
axs[1].set_ylim(0.86, 1.00)

# Indian Pines 子图
axs[2].bar(x - 1.5 * width, indian_baseline, width, label='cos', color=color_baseline)
axs[2].bar(x - 0.5 * width, indian_mean, width, label='mean', color=color_mean)
axs[2].bar(x + 0.5 * width, indian_triangle, width, label='triangle', color=color_triangle)
axs[2].bar(x + 1.5 * width, indian_new, width, label='Ours', color=color_new)
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics)
axs[2].set_title('Indian Pines')
axs[2].legend()
axs[2].set_ylim(0.86, 1.00)

plt.suptitle('Performance Comparison: Different Weighting Methods', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
