import torch


def get_device():
    """
    获取可用的计算设备，优先选择 CUDA，然后是 MPS，最后是 CPU。

    Args:
        None

    Returns:
        torch.device: 可用的计算设备。
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

if __name__ == "__main__":
    print(get_device())