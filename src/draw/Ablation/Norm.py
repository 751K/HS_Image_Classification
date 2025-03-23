import matplotlib.pyplot as plt
import numpy as np


# 指标名称
metrics = ['OA', 'AA', 'Kappa']
x = np.arange(len(metrics))
width = 0.25  # 每个柱子的宽度

color_standard = "#005C53"
color_rms = "#9FC131"
color_none = "#D6D58E"

# Salinas 数据
salinas_OA = [0.9629, 0.9455, 0.9548]
salinas_AA = [0.9295, 0.9034, 0.9407]
salinas_kappa = [0.9506, 0.9274, 0.9400]

salinas_standard = [salinas_OA[0], salinas_AA[0], salinas_kappa[0]]
salinas_rms = [salinas_OA[1], salinas_AA[1], salinas_kappa[1]]
salinas_none = [salinas_OA[2], salinas_AA[2], salinas_kappa[2]]

# Pavia 数据
pavia_OA = [0.9698, 0.9589, 0.9602]
pavia_AA = [0.9534, 0.9360, 0.9554]
pavia_kappa = [0.9599, 0.9454, 0.9471]

pavia_standard = [pavia_OA[0], pavia_AA[0], pavia_kappa[0]]
pavia_rms = [pavia_OA[1], pavia_AA[1], pavia_kappa[1]]
pavia_none = [pavia_OA[2], pavia_AA[2], pavia_kappa[2]]

# Indian Pines 数据
indian_OA = [0.9763, 0.9723, 0.9765]
indian_AA = [0.8871, 0.9064, 0.9003]
indian_kappa = [0.9730, 0.9684, 0.9732]

indian_standard = [indian_OA[0], indian_AA[0], indian_kappa[0]]
indian_rms = [indian_OA[1], indian_AA[1], indian_kappa[1]]
indian_none = [indian_OA[2], indian_AA[2], indian_kappa[2]]

# 创建 1 行 3 列子图，每个子图对应一个数据集
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Salinas 子图
axs[0].bar(x - width, salinas_standard, width, label='Standard', color=color_standard)
axs[0].bar(x, salinas_rms, width, label='RMS', color=color_rms)
axs[0].bar(x + width, salinas_none, width, label='None', color=color_none)
axs[0].set_xticks(x)
axs[0].set_xticklabels(metrics)
axs[0].set_title('Salinas')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_ylim(0.80, 1.00)  # 压缩纵轴范围

# Pavia 子图
axs[1].bar(x - width, pavia_standard, width, label='Standard', color=color_standard)
axs[1].bar(x, pavia_rms, width, label='RMS', color=color_rms)
axs[1].bar(x + width, pavia_none, width, label='None', color=color_none)
axs[1].set_xticks(x)
axs[1].set_xticklabels(metrics)
axs[1].set_title('Pavia')
axs[1].legend()
axs[1].set_ylim(0.80, 1.00)

# Indian Pines 子图
axs[2].bar(x - width, indian_standard, width, label='Standard', color=color_standard)
axs[2].bar(x, indian_rms, width, label='RMS', color=color_rms)
axs[2].bar(x + width, indian_none, width, label='None', color=color_none)
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics)
axs[2].set_title('Indian Pines')
axs[2].legend()
axs[2].set_ylim(0.80, 1.00)

plt.suptitle('Performance Comparison: Different Metrics and Normalization Methods', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()