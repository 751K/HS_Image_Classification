import gc
import torch
import numpy as np
import random
import os


def clear_cache(logger=None):
    """
    清除缓存并重置随机状态，确保试验间的独立性

    Args:
        logger: 可选的日志记录器，用于记录清理过程
    """
    try:
        # 清除CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if logger:
                logger.info("CUDA缓存已清除")

            # 重置CUDA设备状态
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
                if logger:
                    logger.info(f"重置设备 {device} 峰值内存统计")

        # 强制垃圾回收
        collected = gc.collect()
        if logger:
            logger.info(f"垃圾回收完成，收集了{collected}个对象")

        # 再次运行一次垃圾回收，确保彻底清理
        gc.collect()

        # 重置环境变量，确保试验间的独立性
        os.environ['PYTHONHASHSEED'] = str(random.randint(1, 1000))

        if logger:
            logger.info("缓存清理完成，环境重置成功")
    except Exception as e:
        if logger:
            logger.warning(f"清除缓存时发生错误: {str(e)}")
        else:
            print(f"清除缓存时发生错误: {str(e)}")