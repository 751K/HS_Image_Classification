import matplotlib.pyplot as plt
import numpy as np

# 设置风格

# 指标名称
metrics = ['OA', 'AA', 'Kappa']
x = np.arange(len(metrics))
width = 0.25  # 每个柱子的宽度

# Salinas 数据
salinas_baseline = [0.9671, 0.9467, 0.9562]
salinas_cosfusion = [0.9633, 0.9359, 0.9511]
salinas_remove = [0.9635, 0.9360, 0.9514]

# Pavia 数据
pavia_baseline = [0.9698, 0.9534, 0.9599]
pavia_cosfusion = [0.9653, 0.9440, 0.9538]
pavia_remove = [0.9658, 0.9434, 0.9545]

# Indian Pines 数据
indian_baseline = [0.9763, 0.8871, 0.9730]
indian_cosfusion = [0.9752, 0.9012, 0.9718]
indian_remove = [0.9767, 0.8891, 0.9734]

# 自定义颜色
color_baseline = "#005C53"
color_cosfusion = "#9FC131"
color_remove = "#D6D58E"


# 创建 1 行 3 列子图，每个子图对应一个数据集
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Salinas 子图
axs[0].bar(x - width, salinas_baseline, width, label='baseline', color=color_baseline)
axs[0].bar(x, salinas_cosfusion, width, label='cos', color=color_cosfusion)
axs[0].bar(x + width, salinas_remove, width, label='remove', color=color_remove)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics)
axs[0].set_title('Salinas')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_ylim(0.8, 1.00)

# Pavia 子图
axs[1].bar(x - width, pavia_baseline, width, label='baseline', color=color_baseline)
axs[1].bar(x, pavia_cosfusion, width, label='cos', color=color_cosfusion)
axs[1].bar(x + width, pavia_remove, width, label='remove', color=color_remove)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics)
axs[1].set_title('Pavia')
axs[1].legend()
axs[1].set_ylim(0.8, 1.00)

# Indian Pines 子图
axs[2].bar(x - width, indian_baseline, width, label='baseline', color=color_baseline)
axs[2].bar(x, indian_cosfusion, width, label='cos', color=color_cosfusion)
axs[2].bar(x + width, indian_remove, width, label='remove', color=color_remove)
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics)
axs[2].set_title('Indian Pines')
axs[2].legend()
axs[2].set_ylim(0.8, 1.00)

plt.suptitle('Fusion Way Ablation Experiment: Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
