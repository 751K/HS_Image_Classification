# Hyperspectral Image Classification (HSIC)

高光谱图像分类（HSI Classification）项目，实现了多种深度学习模型来分析和分类高光谱遥感图像。项目包含了常用的CNN、Transformer以及最新的Mamba架构的实现。


## 1.使用方法
1. **克隆仓库：**

   ```bash
   git clone https://github.com/751K/HS_Image_Classification.git
   cd HS_Image_Classification
   pip install -r requirements.txt
    ```

2. **运行主程序：**
   ```bash
   python src/main.py
    ```
   
3. **训练模型**
    ```bash
    cd src
    python main.py
    ```

4. **批量训练所有模型**
    ```bash
    cd src
    python run_all.py
    ```
5. **可视化结果**
    ```bash
    cd src
    python vis.py
    ```

## 2.模型支持及路径

| 模型名称              | 类型          | 路径                                      |
|-------------------|-------------|-----------------------------------------|
| ResNet1D          | CNN         | `src.CNNBase.ResNet1D`                  |
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
| SSMamba           | Mamba       | `src.MambaBase.SSMamba`                 |
| MSAFMamba         | Mamba       | `src.MambaBase.MSAFMamba`               |

_模型实现仅供参考，具体的实现细节和参数设置可能会有所不同。请根据实际需求与论文进行对比和调整。_
### 模型拓展
如果需要使用其他模型，请在`src/CNNBase/`、`src/TransformerBase/`、`src/MambaBase/`下添加新的模型实现，并在`src/config.py`中配置使用的模型。
注意，需要在对应的目录的`__init__.py`文件中完成模型导入

## 3.文件说明

   ```
   project/
│
├── datasets/           # 数据集目录
│   ├── Indian/         # Indian Pines 数据集
│   ├── Pavia/          # Pavia University 数据集
│   └── Salinas/        # Salinas 数据集
│
├── results/            # 存储模型训练结果和输出
│
├── src/                # 源代码目录
│   ├── CNNBase/        # 包含各种 CNN 模型的实现
│   ├── MambaBase/      # 包含 Mamba 架构的模型
│   ├── TransformerBase/# 包含 Transformer 模型
│   ├── Dim/            # 降维算法实现
│   ├── Train_and_Eval/ # 训练和评估相关的代码
│   ├── datesets/       # 数据处理相关代码
│   ├── draw/           # 可视化绘图相关代码
│   ├── config.py       # 配置文件
│   ├── main.py         # 主程序入口
│   ├── model_init.py   # 模型初始化代码
│   └── run_all.py      # 批量训练脚本
│
├── LICENSE             # 项目许可证文件
│
└── requirements.txt    # 依赖库
 
 ```
## 4.支持数据集
1. Indian Pines
2. Pavia University
3. Salinas
4. KSC
5. Botswana
### 数据集拓展
如果需要使用其他数据集，请在`src/datesets/datasets_load.py`下添加新的数据集信息，并在`src/config.py`中配置使用的数据集。


## 5.运行环境
已经支持在Linux，MacOS，Windows环境下运行。Windows环境下推荐使用WSL2运行，WSL2具体安装方法请参考[WSL2安装教程](https://docs.microsoft.com/zh-cn/windows/wsl/install)。


## 6.许可证
本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解更多详情。