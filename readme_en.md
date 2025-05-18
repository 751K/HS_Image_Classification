# Hyperspectral Image Classification (HSIC)
[中文 (readme.md)](./readme.md)

## Project Overview & Features

This project focuses on hyperspectral image classification (HSI Classification), implementing a variety of mainstream and cutting-edge deep learning models (CNN, Transformer, Mamba, etc.) for remote sensing HSI analysis and classification. It supports multiple datasets and model comparison experiments, making it suitable for both research and engineering applications.

**Key Features:**
- Supports various mainstream and novel deep learning models
- Clean code structure, easy to extend
- Multi-dataset and batch experiment support
- Visualization and comparative analysis of results

> Note: This project is a graduation design. Some implementations are for reference only. Feedback and suggestions are welcome.

## 1. Usage

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/751K/HS_Image_Classification.git
   cd HS_Image_Classification
   pip install -r requirements.txt
   ```

2. **Run the main program:**
   ```bash
   python src/main.py
   ```

3. **Train a specific model:**
   ```bash
   cd src
   python main.py
   ```

4. **Batch train all models:**
   ```bash
   cd src
   python run_all.py
   ```

5. **Visualize results:**
   ```bash
   cd src
   python vis.py
   ```

## 2. Supported Models & Paths

| Model Name         | Type        | Path                                    |
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

_Implementations are for reference only. Please adjust details and parameters according to papers and actual needs._

### Model Extension
To add a new model:
1. Add your model implementation under `src/CNNBase/`, `src/TransformerBase/`, or `src/MambaBase/`.
2. Import the new model in the corresponding `__init__.py`.
3. Configure the new model in `src/config.py`.

## 3. Project Structure

```
project/
├── datasets/           # Datasets
│   ├── Indian/         # Indian Pines
│   ├── Pavia/          # Pavia University
│   └── Salinas/        # Salinas
├── results/            # Model results and outputs
├── src/                # Source code
│   ├── CNNBase/        # CNN models
│   ├── MambaBase/      # Mamba models
│   ├── TransformerBase/# Transformer models
│   ├── Dim/            # Dimensionality reduction
│   ├── Train_and_Eval/ # Training and evaluation
│   ├── datesets/       # Data processing
│   ├── draw/           # Visualization
│   ├── config.py       # Config file
│   ├── main.py         # Main entry
│   ├── model_init.py   # Model initialization
│   └── run_all.py      # Batch training script
├── LICENSE             # License
└── requirements.txt    # Dependencies
```

## 4. Supported Datasets

- Indian Pines
- Pavia University
- Salinas
- KSC
- Botswana
- WHU-Hi-Honghu

### Dataset Extension
To add a new dataset:
1. Add loading logic in `src/datesets/datasets_load.py`.
2. Configure the new dataset in `src/config.py`.

## 5. Experimental Results

| Model Name        | Dataset       | Accuracy | AA     | Kappa  | Train Time(s) | Params    | Inference Time(s) |
|-------------------|--------------|----------|--------|--------|---------------|-----------|-------------------|
| LeeEtAl3D         | Indian Pines | 0.8022   | 0.7257 | 0.7703 | 119.60        | 158,736   | 0.4582            |
| ResNet3D          | Indian Pines | 0.9470   | 0.8830 | 0.9395 | 130.47        | 418,768   | 0.3068            |
| ResNet2D          | Indian Pines | 0.9871   | 0.9898 | 0.9853 | 559.36        | 11,193,744| 1.6913            |
| GCN2D             | Indian Pines | 0.9026   | 0.6905 | 0.8887 | 22.85         | 23,441    | 0.0570            |
| HybridSN          | Indian Pines | 0.9500   | 0.9231 | 0.9429 | 92.43         | 10,885,248| 0.2625            |
| SwimTransformer   | Indian Pines | 0.9461   | 0.8434 | 0.9385 | 81.72         | 53,264    | 0.2163            |
| VisionTransformer | Indian Pines | 0.9646   | 0.9223 | 0.9596 | 83.04         | 117,264   | 0.2238            |
| SSFTT             | Indian Pines | 0.9376   | 0.8122 | 0.9289 | 20.24         | 165,536   | 0.0287            |
| SFT               | Indian Pines | 0.8323   | 0.6128 | 0.8077 | 13.16         | 136,112   | 0.0197            |
| MambaHSI          | Indian Pines | 0.9429   | 0.8431 | 0.9348 | 862.80        | 105,232   | 2.7553            |
| SSMamba           | Indian Pines | 0.9648   | 0.9598 | 0.9599 | 289.79        | 240,050   | 0.7452            |
| STMamba           | Indian Pines | 0.8473   | 0.6145 | 0.8237 | 602.23        | 69,456    | 1.0509            |
| AllinMamba        | Indian Pines | 0.9879   | 0.9752 | 0.9862 | 417.85        | 2,232,084 | 1.3174            |

## 6. Environment

- Supports Linux, MacOS, Windows (WSL2 recommended)
- Python 3.12+

### Dependencies
1. PyTorch 2.1.0+
2. Numpy
3. scikit-learn
4. See requirements.txt for details

### Hardware Recommendation
1. NVIDIA GPU (RTX 3060 or above recommended)
2. At least 16GB RAM
3. Apple Silicon is also supported, but NVIDIA GPU is recommended

## 7. FAQ

**Q: Error or missing dependencies?**  
A: Please make sure all dependencies in requirements.txt are installed.

**Q: How to customize models or datasets?**  
A: See the "Model Extension" and "Dataset Extension" sections.

**Q: Can I run on CPU?**  
A: Yes, but not recommended.

## 8. Contribution

Issues, PRs, and suggestions are welcome! Please submit issues on GitHub for bugs or improvements.

## 9. Contact

- Author: 751K
- Email: [k1012922528@gmail.com](mailto:k1012922528@gmail.com)

## 10. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 11. Acknowledgement

Thanks to myself and all future contributors.
Thanks to ChatGPT and Claude for their assistance.
