import torch
import time
from mamba_ssm import Mamba2
from mamba_ssm import Mamba

# 定义模型参数
dim = 32  # 模型维度 d_model
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
seq_length = 2000
x = torch.randn(batch_size, seq_length, dim).to("cuda")

# 通过模型进行前向传播
# x = x.contiguous()
start_time = time.time()
y1 = model1(x)
print("Mamba forward time:", time.time() - start_time)
start_time = time.time()
y2 = model2(x)
print("Mamba2 forward time:", time.time() - start_time)
# 第一次需要编译，因此速度较慢，以第二次为准
start_time = time.time()
y3 = model2(x)
print("Mamba2 forward time:", time.time() - start_time)


# 检查输出形状

print("输入形状:", x.shape)
print("输出形状:", y1.shape)
print("输出形状:", y2.shape)
