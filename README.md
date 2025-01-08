# Hyperspectral Image Classification (HSIC)

毕业设计项目，通过模块化组件实现，动态获取模型类别，实现了模型的即插即用，方便用户使用。

## 功能

- 数据预处理
- 特征提取
- 模型训练与评估
- 分类结果可视化

## 技术栈

- 编程语言：Python
- 框架与库：NumPy, SciPy, scikit-learn, PyTorch（根据你使用的库选择）
- 工具：Git, Pycharm

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
