import torch
import time
from mamba_ssm import Mamba2
from mamba_ssm import Mamba

# 定义模型参数
dim = 64  # 模型维度 d_model
d_state = 4  # SSM状态扩展因子
d_conv = 4  # 局部卷积宽度
expand = 4  # 块扩展因子

model1 = Mamba(
    d_model=dim,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand
).to("cuda")

# 对Mamba2，确保 d_model * expand / headdim = 8 的倍数
model2 = Mamba2(
    d_model=dim,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand,
    headdim=16
).to("cuda")

# 生成一些随机输入数据
batch_size = 32
seq_length = 2048
x = torch.randn(batch_size, seq_length, dim).to("cuda")

# 预热运行
print("预热运行...")
with torch.no_grad():
    _ = model1(x)
    _ = model2(x)

# 运行5次并计算平均时间
num_iterations = 100
mamba_times = []
mamba2_times = []

print(f"\n开始{num_iterations}次计时测试...")
with torch.no_grad():
    for i in range(num_iterations):
        x = torch.randn(batch_size, seq_length, dim).to("cuda")
        # Mamba测试
        start_time = time.time()
        y1 = model1(x)
        mamba_times.append(time.time() - start_time)

        # Mamba2测试
        start_time = time.time()
        y3 = model2(x)
        mamba2_times.append(time.time() - start_time)

        print(f"第{i + 1}次运行:")
        print(f"Mamba forward time: {mamba_times[-1]:.4f}s")
        print(f"Mamba2 forward time: {mamba2_times[-1]:.4f}s")

# 计算并打印平均时间
print("\n统计结果:")
print(f"Mamba平均运行时间: {sum(mamba_times)/num_iterations:.4f}s")
print(f"Mamba2平均运行时间: {sum(mamba2_times)/num_iterations:.4f}s")

# 检查输出形状
print("输入形状:", x.shape)
print("输出形状:", y1.shape)
print("输出形状:", y3.shape)
