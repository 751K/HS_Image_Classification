import matplotlib.pyplot as plt

# --------------------
# 1) 预先定义的数据
# --------------------
models = ["ResNet2D", "HybridSN", "LeeEtAl3D", "SSFTT", "SFT", "SSMamba", "MambaHSI", "STMamba", "Ours"]

# 这里的数值仅为示例，请用你自己真实的 FLOPs、Model Size、OA(%) 替换
flops = [120, 200, 350, 500, 620, 700, 900, 1000, 1100]  # 单位：M
parameters = [80, 150, 280, 400, 520, 600, 800, 1000, 1200]  # 单位：K
oa = [94.5, 96.2, 95.8, 97.0, 96.0, 97.2, 98.1, 99.1, 99.3]  # 单位：%

# 为了让不同方法在图上有不同的标记
markers = ["^", "s", "o", "D", "p", "H", "*", "v", "x"]  # 可以自行调整
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]  # 自定义配色
# --------------------
# 2) 创建子图并绘制
# --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# (a) FLOPs vs OA
for i, model in enumerate(models):
    ax1.scatter(flops[i], oa[i],
                marker=markers[i],
                color=colors[i],
                label=model,
                s=100)  # s=100可调整点的大小
ax1.set_xlabel("FLOPs (M)")
ax1.set_ylabel("OA (%)")
ax1.set_title("(a)")

# (b) Model Parameters vs OA
for i, model in enumerate(models):
    ax2.scatter(parameters[i], oa[i],
                marker=markers[i],
                color=colors[i],
                label=model,
                s=100)
ax2.set_xlabel("Model Parameters (M)")
ax2.set_ylabel("OA (%)")
ax2.set_title("(b)")

# --------------------
# 3) 设置图例
# --------------------
ax1.legend(loc="lower right")
ax1.grid(True)
ax2.legend(loc="lower right")
ax2.grid(True)

# --------------------
# 4) 布局与显示
# --------------------
plt.tight_layout()
plt.show()
