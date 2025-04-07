import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取CSV数据
df = pd.read_csv('model_comodel_comparison_Indian.csvmparison_Indian.csv')

# 获取模型名称列表
models = df['models'].tolist()

# 准备数据 (将准确率转换为百分比)
parameters = [p / 1000 for p in df['parameters'].tolist()]  # 转换为K单位
oa = [acc * 100 for acc in df['accuracy'].tolist()]  # 转换为百分比
inference_time = df['inference_time'].tolist()  # 推理时间(ms)

# 自动生成足够的标记和颜色
markers = ["o", "s", "^", "D", "p", "H", "*", "v", "x", "+", "d", "<", ">"]
colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# (a) 推理时间 vs OA
for i, model in enumerate(models):
    ax1.scatter(inference_time[i], oa[i],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=model,
                s=100)
    # 在每个点旁添加模型名称标签
    ax1.annotate(model, (inference_time[i], oa[i]),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=8, color=colors[i])

ax1.set_xlabel("Inference time (ms/sample)")
ax1.set_ylabel("OA (%)")
ax1.set_title("(a) Inference time vs Accuracy")
ax1.grid(True)

# (b) 参数量 vs OA
for i, model in enumerate(models):
    ax2.scatter(parameters[i], oa[i],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=model,
                s=100)
    # 在每个点旁添加模型名称标签
    ax2.annotate(model, (parameters[i], oa[i]),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=8, color=colors[i])

ax2.set_xlabel("Model Parameters (K)")
ax2.set_ylabel("OA (%)")
ax2.set_title("(b) Parameters vs Accuracy")
ax2.set_xscale('log')  # 参数量使用对数刻度更合适
ax2.grid(True)

# 添加常规图例，位于底部
fig.legend(handles=[plt.Line2D([0], [0], marker=markers[i % len(markers)], color=colors[i],
                              linestyle='None', markersize=8, label=model)
                   for i, model in enumerate(models)],
          loc='upper center', bbox_to_anchor=(0.5, 0),
          ncol=5, fancybox=True, shadow=True, fontsize='medium')

# 布局与保存
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为底部图例留出更多空间
plt.savefig("model_performance_comparison.png", dpi=300, bbox_inches="tight")
plt.show()