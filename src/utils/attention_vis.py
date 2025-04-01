# src/utils/attention_vis.py
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import cv2
from einops import rearrange


class AttentionVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = []
        self.attention_maps = {}
        self.feature_maps = {}
        self.gradients = {}

    def register_hooks(self):
        """为模型中的关键组件注册钩子函数"""
        # SSD选择性注意力可视化
        if hasattr(self.model, 'MambaLayer1'):
            self._register_ssd_hook('MambaLayer1')
        if hasattr(self.model, 'MambaLayer2'):
            self._register_ssd_hook('MambaLayer2')

        # 融合门控机制可视化
        if hasattr(self.model, 'fusion_gate'):
            handle = self.model.fusion_gate.register_forward_hook(self._fusion_gate_hook)
            self.hooks.append(handle)

        # 扫描权重可视化
        if hasattr(self.model, 'scan_way'):
            handle = self.model.register_forward_hook(self._scan_hook)
            self.hooks.append(handle)

        # 特征图可视化 - 用于类激活映射
        # 修正：在conv2d_sep模块上注册钩子，而不是在feature_extraction方法上
        if hasattr(self.model, 'conv2d_sep'):
            handle = self.model.conv2d_sep.register_forward_hook(self._feature_hook)
            self.hooks.append(handle)

    def _register_ssd_hook(self, layer_name):
        """为SSD模块注册钩子"""

        def ssd_hook(module, inputs, output):
            # 捕获关键注意力变量
            if hasattr(module, 'in_proj'):
                self.attention_maps[f"{layer_name}_projections"] = module.in_proj.weight.detach().cpu()

            # 获取A值 - Mamba2特有
            if hasattr(module, 'A_log'):
                self.attention_maps[f"{layer_name}_A"] = module.A_log.detach().cpu()

            # 获取D值 - Mamba2特有
            if hasattr(module, 'D'):
                self.attention_maps[f"{layer_name}_D"] = module.D.detach().cpu()

        layer = getattr(self.model, layer_name)
        handle = layer.register_forward_hook(ssd_hook)
        self.hooks.append(handle)

    def _fusion_gate_hook(self, module, inputs, outputs):
        """捕获融合门权重"""
        self.attention_maps['fusion_weights'] = outputs.detach().cpu()

    def _scan_hook(self, module, inputs, outputs):
        """捕获扫描权重"""
        if hasattr(module, '_weights_cache') and module._weights_cache:
            self.attention_maps['scan_weights'] = module._weights_cache.copy()

    def _feature_hook(self, module, inputs, outputs):
        """捕获特征图"""
        self.feature_maps['features'] = outputs.detach()

        # 注册梯度钩子（只在训练模式下）
        if outputs.requires_grad:
            outputs.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        """捕获梯度"""
        self.gradients['features'] = grad.detach()

    @staticmethod
    def _save_to_csv(data_tensor, file_path):
        """辅助函数：保存二维 tensor 到 CSV"""
        # 仅处理二维数据
        if data_tensor.dim() == 2:
            df = pd.DataFrame(data_tensor.numpy())
            df.to_csv(file_path, index=False)

    def visualize_attention(self, input_data, save_dir, class_names=None):
        """执行前向传播并可视化注意力权重"""
        self.model.eval()
        input_data = input_data.to(self.device)

        # 清除之前的注意力图
        self.attention_maps = {}
        self.feature_maps = {}

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 执行前向传播
        with torch.no_grad():
            outputs = self.model(input_data)

        # 保存注意力相关数据到 CSV（仅保存二维数据）
        for name, tensor in self.attention_maps.items():
            csv_file = os.path.join(save_dir, f"{name}.csv")
            if tensor.dim() == 2:
                self._save_to_csv(tensor, csv_file)

        # 可视化融合门权重
        if 'fusion_weights' in self.attention_maps:
            self._plot_fusion_weights(save_dir)

        # 可视化扫描权重
        if 'scan_weights' in self.attention_maps:
            self._plot_scan_weights(save_dir)

        # 可视化SSD注意力
        for name in self.attention_maps:
            if 'projections' in name or 'A' in name or 'D' in name:
                self._plot_ssd_attention(name, save_dir)

        # 生成类激活映射
        if 'features' in self.feature_maps:
            self.generate_cam(input_data, save_dir)

    def _plot_fusion_weights(self, save_dir):
        """绘制融合门权重"""
        weights = self.attention_maps['fusion_weights']
        batch_size = weights.shape[0]

        plt.figure(figsize=(10, 6))
        # 选择最多16个样本可视化
        n_samples = min(16, batch_size)

        # 创建热力图
        for i in range(n_samples):
            plt.subplot(4, 4, i + 1 if i < 16 else 16)
            w = weights[i].numpy().reshape(1, -1)
            sns.heatmap(w, cmap='viridis', annot=True, fmt='.2f', cbar=False)
            plt.title(f'Sample {i + 1}')
            plt.xlabel('Spatial | Spectral')
            plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fusion_weights.png'), dpi=300)
        plt.close()

    def _plot_scan_weights(self, save_dir):
        """绘制扫描权重"""
        weights_cache = self.attention_maps['scan_weights']

        plt.figure(figsize=(15, 10))
        idx = 1
        for size, weights in weights_cache.items():
            if idx > 16:  # 最多绘制16个权重
                break

            ax = plt.subplot(4, 4, idx)
            idx += 1

            # 将权重重塑为二维图像
            if size > 1:
                weights_2d = weights.reshape(size, size)
                sns.heatmap(weights_2d, ax=ax, cmap='viridis')
            else:
                ax.bar(range(len(weights)), weights.numpy())

            ax.set_title(f'Size {size}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Weight')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'scan_weights.png'), dpi=300)
        plt.close()

    def _plot_ssd_attention(self, name, save_dir):
        """绘制SSD注意力映射"""
        outputs = self.attention_maps[name]

        # 选择第一个批次的输出
        if outputs.dim() >= 3:
            # 取中间层输出的平均幅值
            attention = outputs[0].abs().mean(dim=-1) if outputs.dim() > 3 else outputs[0]
            plt.figure(figsize=(10, 6))

            # 确保数据是二维的并处理掩码问题
            if attention.dim() == 1:
                # 如果是一维的，重塑为二维数组
                attention_2d = attention.unsqueeze(0)  # 添加一个维度，变成(1, seq_len)
                sns.heatmap(attention_2d.numpy(), cmap='viridis', mask=None)
            else:
                # 已经是二维了
                sns.heatmap(attention.numpy(), cmap='viridis', mask=None)

            plt.title(f'{name} Attention Distribution')
            plt.xlabel('Sequence Position')
            plt.ylabel('Head')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{name}_attention.png'), dpi=300)
            plt.close()
        elif outputs.dim() == 1:
            # 处理一维张量
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(outputs)), outputs.numpy())
            plt.title(f'{name} Value Distribution')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{name}_values.png'), dpi=300)
            plt.close()

    def generate_cam(self, input_data, save_dir, target_class=None, alpha=0.5):
        """生成类激活映射 (CAM)"""
        if 'features' not in self.feature_maps:
            print("Warning: No feature maps captured, skipping CAM generation")
            return

        # 获取特征
        features = self.feature_maps['features']
        batch_size = features.shape[0]

        # 如果没有梯度信息，使用特征图的平均值作为权重
        if not hasattr(self, 'gradients') or 'features' not in self.gradients:
            print("Note: Using feature average for CAM generation (no gradients)")
            # 计算特征图的通道平均值
            cam = features.mean(dim=1)  # (batch, H, W)
        else:
            # 使用梯度加权特征图
            gradients = self.gradients['features']
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)

            # 计算加权和
            cam = torch.zeros(batch_size, features.shape[2], features.shape[3], device=features.device)
            for i in range(batch_size):
                for c in range(features.shape[1]):
                    cam[i] += weights[i, c, 0, 0] * features[i, c]

        # 应用ReLU并归一化
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)

        # 可视化CAM (最多16个样本)
        n_samples = min(16, batch_size)
        plt.figure(figsize=(20, 20))

        for i in range(n_samples):
            # 将输入图像转换为可视化格式
            if input_data.dim() == 4:  # (B, C, H, W)
                # 高光谱图像通常需要特殊处理
                # 可视化选择的三个通道作为RGB
                c = input_data.shape[1]
                if c >= 3:
                    # 选择三个较为平均分布的通道
                    idx1, idx2, idx3 = c // 4, c // 2, c * 3 // 4
                    rgb_img = torch.stack([
                        input_data[i, idx1],
                        input_data[i, idx2],
                        input_data[i, idx3]
                    ], dim=-1).cpu().numpy()
                else:
                    # 单通道重复
                    rgb_img = input_data[i, 0:1].repeat(1, 3).permute(1, 2, 0).cpu().numpy()

                # 归一化图像
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-7)

                # CAM热力图
                heatmap = cam[i].cpu().numpy()
                heatmap = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

                # 叠加
                superimposed = alpha * heatmap + (1 - alpha) * rgb_img

                plt.subplot(4, 4, i + 1)
                plt.imshow(superimposed)
                plt.title(f'Sample {i + 1}')
                plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_activation_maps.png'), dpi=300)
        plt.close()

    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


# python
def visualize_mamba_attention(model, dataloader, device, save_dir, class_names=None, n_samples=16):
    """Mamba注意力可视化主函数"""
    os.makedirs(save_dir, exist_ok=True)
    visualizer = AttentionVisualizer(model, device)
    visualizer.register_hooks()
    for inputs, labels in dataloader:
        if inputs.shape[0] > n_samples:
            inputs = inputs[:n_samples]
            labels = labels[:n_samples]
        batch_dir = os.path.join(save_dir, 'batch_attention')
        visualizer.visualize_attention(inputs.to(device), batch_dir, class_names)
        if class_names:
            unique_labels = torch.unique(labels)
            for label in unique_labels:
                idx = (labels == label).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    class_dir = os.path.join(save_dir, f'class_{label.item()}_{class_names[label.item()]}')
                    os.makedirs(class_dir, exist_ok=True)
                    visualizer.visualize_attention(inputs[idx[:1]].to(device), class_dir)
        break
    visualizer.remove_hooks()