# Hyperspectral Image Classification (HSIC)

毕业设计项目，通过模块化组件实现，动态获取模型类别，实现了模型的即插即用，方便用户使用。

## 功能

- 数据预处理
- 特征提取
- 模型训练与评估
- 分类结果可视化

## 模型类型及路径

| 模型名称              | 类型          | 路径                                      |
|-------------------|-------------|-----------------------------------------|
| ResNet1D          | CNN         | `src.CNNBase.ResNet1D`                  |
| ResNet2D          | CNN         | `src.CNNBase.ResNet2D`                  |
| ResNet3D          | CNN         | `src.CNNBase.ResNet3D`                  |
| HybridSN          | CNN         | `src.CNNBase.HybridSN`                  |
| LeeEtAl3D         | CNN         | `src.CNNBase.LeeEtAl3D`                 |
| GCM2D             | GCN         | `src.CNNBase.GCN2D`                     |
| SSFTT             | Transformer | `src.TransformerBase.SSFTT`             |
| SwimTransformer   | Transformer | `src.TransformerBase.SwimTransformer`   |
| VisionTransformer | Transformer | `src.TransformerBase.VisionTransformer` |
| SFT               | Transformer | `src.TransformerBase.SFT`               |
| SSMamba           | Mamba       | `src.MambaBase.SSMamba`                 |
| MambaHSI          | Mamba       | `src.MambaBase.MambaHSI`                |
| STMamba           | Mamba       | `src.MambaBase.STMamba`                 |
| SSMamba           | Mamba       | `src.MambaBase.SSMamba`                 |
| MSAFMamba         | Mamba       | `src.MambaBase.MSAFMamba`               |




## 使用说明

1. 克隆仓库：

   ```bash
   git clone https://github.com/751K/HS_Image_Classification.git
    ```

2. 运行主程序：

   ```bash
   python src/main.py
    ```
## 文件说明

- `results/`: 存储模型训练结果和输出
- `src/`: 源代码目录
  - `CNNBase/`: 包含各种 CNN 模型的实现
  - `datasets/`: 数据集处理相关代码
  - `Train_and_Eval/`: 训练和评估相关的代码
  - `config.py`: 配置文件
  - `main.py`: 主程序入口
  - `model_init.py`: 模型初始化代码
- `LICENSE`: 项目许可证文件
- `README.md`: 项目说明文档

## 运行环境
由于代码中引入了Mamba模型，代码在Windows环境下使用WSL2运行，WSL2具体安装方法请参考[WSL2安装教程](https://docs.microsoft.com/zh-cn/windows/wsl/install)。
- pytorch:2.22
- python:3.12
- CUDA:11.8