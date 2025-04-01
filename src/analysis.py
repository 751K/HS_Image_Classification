import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from thop import profile, clever_format
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
from datetime import datetime

from src.MambaBase.Mamba2 import Mamba2
from src.MambaBase.AllinMamba import AllinMamba
from src.utils.device import get_device
from src.utils.paths import ROOT_DIR, ensure_dir, sanitize_filename

# 创建模型分析专用目录
ANALYSIS_DIR = os.path.join(ROOT_DIR, "model_analysis")
ensure_dir(ANALYSIS_DIR)


def count_parameters(model):
    """计算模型的参数量"""
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    return {
        'total': total_params,
        'trainable': trainable_params
    }


def analyze_parameters_by_layer(model):
    """分析模型每一层的参数量"""
    results = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只统计叶子模块
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                results[name] = param_count
    return results


def measure_throughput(model, input_shape, batch_sizes=[1, 2, 4, 8, 16, 32, 64], device='cuda', warmup=10, repeats=100):
    """测量不同批大小下的吞吐量"""
    model.eval()
    results = {}

    for batch_size in batch_sizes:
        # 创建输入数据
        if isinstance(input_shape, tuple) and len(input_shape) == 3:  # (C, H, W)
            input_data = torch.randn(batch_size, *input_shape).to(device)
        else:
            raise ValueError("input_shape应该是(C, H, W)格式")

        # 预热
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_data)

        # 计时
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        for _ in range(repeats):
            with torch.no_grad():
                _ = model(input_data)

        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()

        # 计算吞吐量
        elapsed_time = end_time - start_time
        samples_per_second = batch_size * repeats / elapsed_time
        inference_time = elapsed_time / repeats

        results[batch_size] = {
            'inference_time': inference_time,
            'throughput': samples_per_second
        }

    return results


def calculate_flops(model, input_shape, device='cuda'):
    """计算模型的FLOPs"""
    if isinstance(input_shape, tuple) and len(input_shape) == 3:  # (C, H, W)
        input_data = torch.randn(1, *input_shape).to(device)
    else:
        raise ValueError("input_shape应该是(C, H, W)格式")

    macs, params = profile(model, inputs=(input_data,))
    flops = macs * 2  # 一个MAC大约对应2个FLOPs

    flops_human, params_human = clever_format([flops, params], "%.3f")

    return {
        'flops': flops,
        'flops_human': flops_human,
        'params': params,
        'params_human': params_human
    }


def export_results(parameters, layer_parameters, throughput, flops, output_dir):
    """导出分析结果到文本和图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存参数统计结果
    with open(os.path.join(output_dir, 'model_analysis.txt'), 'w') as f:
        f.write(f"模型总参数量: {parameters['total']:,}\n")
        f.write(f"可训练参数量: {parameters['trainable']:,}\n\n")

        f.write(f"计算量统计:\n")
        f.write(f"总FLOPs: {flops['flops_human']}\n")
        f.write(f"总参数数量: {flops['params_human']}\n\n")

        f.write("各层参数统计:\n")
        for name, params in sorted(layer_parameters.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{name}: {params:,}\n")

        f.write("\n吞吐量统计:\n")
        for batch_size, metrics in throughput.items():
            f.write(f"Batch Size {batch_size}:\n")
            f.write(f"  推理时间: {metrics['inference_time'] * 1000:.2f} ms/batch\n")
            f.write(f"  吞吐量: {metrics['throughput']:.2f} samples/sec\n")

    # 生成吞吐量图表
    batch_sizes = list(throughput.keys())
    throughputs = [metrics['throughput'] for metrics in throughput.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, throughputs, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('Model Throughput vs. Batch Size')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'throughput.png'))

    # 生成层参数量柱状图（只显示参数量最多的前15层）
    top_layers = dict(sorted(layer_parameters.items(), key=lambda x: x[1], reverse=True)[:15])

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(top_layers)), list(top_layers.values()))
    plt.xticks(range(len(top_layers)), list(top_layers.keys()), rotation=90)
    plt.ylabel('Parameters Count')
    plt.title('Top 15 Layers by Parameter Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_parameters.png'))


def create_analysis_dir(model_name, channels, patch_size):
    """创建模型分析结果目录"""
    timestamp = datetime.now().strftime('%m%d_%H%M')
    folder_name = f"{model_name}_c{channels}_p{patch_size}_{timestamp}"
    save_dir = os.path.join(ANALYSIS_DIR, folder_name)
    return ensure_dir(save_dir)


def analyze_model(model_name, model_params, input_shape, output_dir=None, batch_sizes=[1, 2, 4, 8, 16, 32, 64]):
    """分析模型并导出结果"""
    device = get_device()
    print(f"使用设备: {device}")

    # 如果未提供输出目录，创建默认目录
    if output_dir is None:
        output_dir = create_analysis_dir(model_name,
                                         model_params.get('input_channels', 0),
                                         model_params.get('patch_size', 0))
    else:
        # 确保提供的目录存在
        output_dir = ensure_dir(output_dir)

    print(f"加载模型: {model_name}")
    if model_name == 'AllinMamba':
        model = AllinMamba(**model_params).to(device)
    elif model_name == 'Mamba2':
        model = Mamba2(**model_params).to(device)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    print("计算参数量...")
    parameters = count_parameters(model)
    layer_parameters = analyze_parameters_by_layer(model)

    print("计算FLOPs...")
    flops = calculate_flops(model, input_shape, device)

    print("测量吞吐量...")
    throughput = measure_throughput(model, input_shape, batch_sizes, device)

    print("导出分析结果...")
    export_results(parameters, layer_parameters, throughput, flops, output_dir)

    print(f"分析完成，结果已保存到: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型分析工具')
    parser.add_argument('--model', type=str, default='AllinMamba', choices=['AllinMamba', 'Mamba2'],
                        help='要分析的模型名称')
    parser.add_argument('--channels', type=int, default=80, help='输入通道数')
    parser.add_argument('--patch_size', type=int, default=9, help='patch大小')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，若不指定则自动生成')
    parser.add_argument('--max_batch', type=int, default=64, help='最大批大小')
    parser.add_argument('--depth', type=int, default=1, help='AllinMamba的深度')
    parser.add_argument('--d_model', type=int, default=512, help='Mamba2的模型维度')
    parser.add_argument('--d_state', type=int, default=64, help='Mamba2的状态维度')

    args = parser.parse_args()

    # 设置模型参数
    if args.model == 'AllinMamba':
        model_params = {
            'input_channels': args.channels,
            'num_classes': 10,
            'patch_size': args.patch_size,
            'depth': args.depth
        }
    elif args.model == 'Mamba2':
        model_params = {
            'd_model': args.d_model,
            'd_state': args.d_state,
            'headdim': 64,
            'chunk_size': 8,
            'expand': 2
        }

    input_shape = (args.channels, args.patch_size, args.patch_size)
    batch_sizes = [1, 2, 4, 8, 16, 32, args.max_batch]

    # 如果指定了输出目录
    output_dir = args.output
    if output_dir is None:
        # 使用自动生成的目录
        pass
    elif not os.path.isabs(output_dir):
        # 如果是相对路径，将其放在ANALYSIS_DIR下
        output_dir = os.path.join(ANALYSIS_DIR, output_dir)

    analyze_model(args.model, model_params, input_shape, output_dir, batch_sizes)
