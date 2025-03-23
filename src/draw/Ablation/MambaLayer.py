import matplotlib.pyplot as plt
import numpy as np

# 指标名称
metrics = ['OA', 'AA', 'Kappa']
x = np.arange(len(metrics))
width = 0.2  # 每个柱子的宽度

# 自定义颜色
color_baseline = "#005C53"
color_remove1 = "#9FC131"
color_remove2 = "#DBF227"
color_remove3 = "#D6D58E"

# Salinas 数据
salinas_baseline = [0.9671, 0.9467, 0.9562]
salinas_layer1 = [0.9339, 0.9023, 0.9114]
salinas_layer2 = [0.8054, 0.7050, 0.7361]
salinas_layer3 = [0.9664, 0.9517, 0.9554]

# Pavia 数据
pavia_baseline = [0.9698, 0.9534, 0.9599]
pavia_layer1 = [0.9187, 0.8652, 0.8909]
pavia_layer2 = [0.8054, 0.7050, 0.7361]
pavia_layer3 = [0.9646, 0.9407, 0.9529]

# Indian Pines 数据
indian_baseline = [0.9763, 0.8871, 0.9730]
indian_layer1 = [0.9355, 0.7791, 0.9262]
indian_layer2 = [0.7037, 0.5427, 0.6552]
indian_layer3 = [0.9512, 0.7658, 0.9444]

# 创建 1 行 3 列子图，每个子图对应一个数据集
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Salinas 子图
axs[0].bar(x - 1.5 * width, salinas_baseline, width, label='baseline', color=color_baseline)
axs[0].bar(x - 0.5 * width, salinas_layer1, width, label='-Layer1', color=color_remove1)
axs[0].bar(x + 0.5 * width, salinas_layer2, width, label='-Layer2', color=color_remove2)
axs[0].bar(x + 1.5 * width, salinas_layer3, width, label='-Layer3', color=color_remove3)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics)
axs[0].set_title('Salinas')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_ylim(0.50, 1.00)  # 压缩纵轴范围

# Pavia 子图
axs[1].bar(x - 1.5 * width, pavia_baseline, width, label='baseline', color=color_baseline)
axs[1].bar(x - 0.5 * width, pavia_layer1, width, label='-Layer1', color=color_remove1)
axs[1].bar(x + 0.5 * width, pavia_layer2, width, label='-Layer2', color=color_remove2)
axs[1].bar(x + 1.5 * width, pavia_layer3, width, label='-Layer3', color=color_remove3)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics)
axs[1].set_title('Pavia')
axs[1].legend()
axs[1].set_ylim(0.50, 1.00)

# Indian Pines 子图
axs[2].bar(x - 1.5 * width, indian_baseline, width, label='baseline', color=color_baseline)
axs[2].bar(x - 0.5 * width, indian_layer1, width, label='-Layer1', color=color_remove1)
axs[2].bar(x + 0.5 * width, indian_layer2, width, label='-Layer2', color=color_remove2)
axs[2].bar(x + 1.5 * width, indian_layer3, width, label='-Layer3', color=color_remove3)
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics)
axs[2].set_title('Indian Pines')
axs[2].legend()
axs[2].set_ylim(0.50, 1.00)

plt.suptitle('MambaLayer Ablation Experiment: Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
