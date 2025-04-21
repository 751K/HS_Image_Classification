import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体，Mac上常用的中文字体
try:
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
    plt.rcParams['font.family'] = ['Arial Unicode MS']
except:
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据，并转置
file_name = 'model_comparison_Botswana_25.csv'
df = pd.read_csv(file_name, index_col=0)

# 获取模型名称列表
models = df.index.tolist()

# 创建图例名称映射字典
legend_names = {model: model for model in models}
legend_names["AllinMamba"] = "ours"  # 将AllinMamba在图例中显示为ours

# 准备数据 (将准确率转换为百分比)
parameters = [float(p) / 1000 for p in df['parameters'].tolist()]  # 转换为K单位
oa = [float(acc) * 100 for acc in df['accuracy'].tolist()]  # 转换为百分比
inference_time = [float(t) for t in df['inference_time'].tolist()]  # 推理时间(ms)

# 自动生成足够的标记和颜色
markers = ["o", "s", "^", "D", "p", "H", "*", "v", "x", "+", "d", "<", ">"]
base_colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
colors = []
for color in base_colors:
    hsv_color = mpl.colors.rgb_to_hsv(color[:3])
    hsv_color[1] = min(hsv_color[1] * 3, 1.0)  # 增加饱和度
    rgb_color = mpl.colors.hsv_to_rgb(hsv_color)
    colors.append(np.append(rgb_color, color[3]))
colors = np.array(colors)

# 图1：推理时间 vs OA
plt.figure(figsize=(10, 8))
for i, model in enumerate(models):
    is_highlight = model == "AllinMamba"
    plt.scatter(inference_time[i], oa[i],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=legend_names[model],
                s=120 if is_highlight else 100)
    # 在每个点旁添加模型名称标签
    display_name = "ours" if model == "AllinMamba" else model
    font_size = 14 if is_highlight else 8
    plt.annotate(display_name, (inference_time[i], oa[i]),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=font_size, color=colors[i],
                 weight='bold' if is_highlight else 'normal')

plt.xlabel("推理速度 (ms/sample)", fontsize=12)
plt.ylabel("OA (%)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize='medium')
plt.tight_layout()

plt.savefig("inference_time_vs_accuracy.pdf", format='pdf', bbox_inches="tight")
plt.savefig("inference_time_vs_accuracy.svg", format='svg', bbox_inches="tight")
plt.show()

# 图2：参数量 vs OA
plt.figure(figsize=(10, 8))
for i, model in enumerate(models):
    is_highlight = model == "AllinMamba"
    plt.scatter(parameters[i], oa[i],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=legend_names[model],
                s=120 if is_highlight else 100)
    # 在每个点旁添加模型名称标签
    display_name = "ours" if model == "AllinMamba" else model
    font_size = 13 if is_highlight else 9
    plt.annotate(display_name, (parameters[i], oa[i]),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=font_size, color=colors[i],
                 weight='bold' if is_highlight else 'normal')

plt.xlabel("模型参数量 (K)", fontsize=12)
plt.ylabel("OA (%)", fontsize=12)
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', fancybox=True, shadow=True, fontsize='medium')
plt.tight_layout()

plt.savefig("parameters_vs_accuracy.svg", format='svg', bbox_inches="tight")
plt.show()