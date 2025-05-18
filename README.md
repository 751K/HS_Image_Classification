
# Hyperspectral Image Classification (HSIC)
[English Version (readme_en.md)](./readme_en.md)

## 项目简介与特色

本项目致力于高光谱图像分类（HSI Classification），实现了多种主流及前沿深度学习模型（CNN、Transformer、Mamba等）用于遥感高光谱图像的分析与分类。支持多数据集、多模型对比实验，便于科研与工程应用。

**项目特色：**
- 支持多种主流与新型深度学习模型
- 代码结构清晰，易于扩展
- 支持多数据集与批量实验
- 结果可视化与对比分析

> 说明：本项目为毕业设计，部分实现仅供参考，欢迎指正与交流。

## 1. 使用方法

1. **克隆仓库并安装依赖：**
   ```bash
   git clone https://github.com/751K/HS_Image_Classification.git
   cd HS_Image_Classification
   pip install -r requirements.txt
   ```

2. **运行主程序：**
   ```bash
   python src/main.py
   ```

3. **训练指定模型：**
   ```bash
   cd src
   python main.py
   ```

4. **批量训练所有模型：**
   ```bash
   cd src
   python run_all.py
   ```

5. **可视化结果：**
   ```bash
   cd src
   python vis.py
   ```

## 2. 模型支持及路径

| 模型名称              | 类型          | 路径                                      |
|-------------------|-------------|-----------------------------------------|
| ResNet2D          | CNN         | `src.CNNBase.ResNet2D`                  |
| ResNet3D          | CNN         | `src.CNNBase.ResNet3D`                  |
| HybridSN          | CNN         | `src.CNNBase.HybridSN`                  |
| LeeEtAl3D         | CNN         | `src.CNNBase.LeeEtAl3D`                 |
| GCN2D             | GCN         | `src.CNNBase.GCN2D`                     |
| SSFTT             | Transformer | `src.TransformerBase.SSFTT`             |
| SwimTransformer   | Transformer | `src.TransformerBase.SwimTransformer`   |
| VisionTransformer | Transformer | `src.TransformerBase.VisionTransformer` |
| SFT               | Transformer | `src.TransformerBase.SFT`               |
| SSMamba           | Mamba       | `src.MambaBase.SSMamba`                 |
| MambaHSI          | Mamba       | `src.MambaBase.MambaHSI`                |
| STMamba           | Mamba       | `src.MambaBase.STMamba`                 |
| AllinMamba        | Mamba       | `src.MambaBase.AllinMamba`              |

_模型实现仅供参考，具体实现细节和参数设置请结合论文与实际需求调整。_

### 模型扩展
如需添加新模型：
1. 在`src/CNNBase/`、`src/TransformerBase/`、`src/MambaBase/`目录下添加模型实现。
2. 在对应目录的`__init__.py`中导入新模型。
3. 在`src/config.py`中配置新模型。

## 3. 文件结构说明

```
project/
├── datasets/           # 数据集目录
│   ├── Indian/         # Indian Pines 数据集
│   ├── Pavia/          # Pavia University 数据集
│   └── Salinas/        # Salinas 数据集
├── results/            # 存储模型训练结果和输出
├── src/                # 源代码目录
│   ├── CNNBase/        # CNN模型实现
│   ├── MambaBase/      # Mamba模型实现
│   ├── TransformerBase/# Transformer模型实现
│   ├── Dim/            # 降维算法
│   ├── Train_and_Eval/ # 训练与评估
│   ├── datesets/       # 数据处理
│   ├── draw/           # 可视化
│   ├── config.py       # 配置文件
│   ├── main.py         # 主程序入口
│   ├── model_init.py   # 模型初始化
│   └── run_all.py      # 批量训练脚本
├── LICENSE             # 许可证
└── requirements.txt    # 依赖库
```

## 4. 支持数据集

- Indian Pines
- Pavia University
- Salinas
- KSC
- Botswana
- WHU-Hi-Honghu

### 数据集扩展
如需添加新数据集：
1. 在`src/datesets/datasets_load.py`中添加数据集加载逻辑。
2. 在`src/config.py`中配置新数据集。

## 5. 实验结果

| 模型名称              | 数据集          | 准确率    | AA     | Kappa  | 训练时长(s) | 参数量        | 推理时长(s) |
|-------------------|--------------|--------|--------|--------|---------|------------|---------|
| LeeEtAl3D         | Indian Pines | 0.8022 | 0.7257 | 0.7703 | 119.60  | 158,736    | 0.4582  |
| ResNet3D          | Indian Pines | 0.9470 | 0.8830 | 0.9395 | 130.47  | 418,768    | 0.3068  |
| ResNet2D          | Indian Pines | 0.9871 | 0.9898 | 0.9853 | 559.36  | 11,193,744 | 1.6913  |
| GCN2D             | Indian Pines | 0.9026 | 0.6905 | 0.8887 | 22.85   | 23,441     | 0.0570  |
| HybridSN          | Indian Pines | 0.9500 | 0.9231 | 0.9429 | 92.43   | 10,885,248 | 0.2625  |
| SwimTransformer   | Indian Pines | 0.9461 | 0.8434 | 0.9385 | 81.72   | 53,264     | 0.2163  |
| VisionTransformer | Indian Pines | 0.9646 | 0.9223 | 0.9596 | 83.04   | 117,264    | 0.2238  |
| SSFTT             | Indian Pines | 0.9376 | 0.8122 | 0.9289 | 20.24   | 165,536    | 0.0287  |
| SFT               | Indian Pines | 0.8323 | 0.6128 | 0.8077 | 13.16   | 136,112    | 0.0197  |
| MambaHSI          | Indian Pines | 0.9429 | 0.8431 | 0.9348 | 862.80  | 105,232    | 2.7553  |
| SSMamba           | Indian Pines | 0.9648 | 0.9598 | 0.9599 | 289.79  | 240,050    | 0.7452  |
| STMamba           | Indian Pines | 0.8473 | 0.6145 | 0.8237 | 602.23  | 69,456     | 1.0509  |
| AllinMamba        | Indian Pines | 0.9879 | 0.9752 | 0.9862 | 417.85  | 2,232,084  | 1.3174  |

## 6. 运行环境

- 支持 Linux、MacOS、Windows（推荐WSL2）
- Python 3.12 及以上

### 依赖库
1. PyTorch 2.1.0 及以上
2. Numpy
3. scikit-learn
4. 详细见 requirements.txt

### 硬件推荐
1. NVIDIA GPU（推荐 RTX 3060 及以上）
2. 至少 16GB 内存
3. Apple Silicon 也不是不能用

## 7. 常见问题（FAQ）

**Q: 运行报错/缺少依赖？**  
A: 请先确保已正确安装 requirements.txt 中所有依赖。

**Q: 如何自定义模型或数据集？**  
A: 参考“模型扩展”与“数据集扩展”部分。

**Q: 可以在CPU上运行吗？**  
A: 可以，但不推荐。

## 8. 贡献指南

欢迎提交 issue、PR 或建议！如有 bug 或改进建议请在 GitHub 仓库提交 issue。

## 9. 联系方式

- 作者：751K
- 邮箱：[k1012922528@gmail.com](mailto:k1012922528@gmail.com)

## 10. 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 11. 致谢

感谢本人以及未来所有的开发者
感谢尊敬的ChatGPT与Claude大人

