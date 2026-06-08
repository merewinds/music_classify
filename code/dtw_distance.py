"""
DTW (Dynamic Time Warping) 距离实现

使用 dtaidistance 库（C 加速）进行快速 DTW 计算
对 1D pitch 序列做对齐，支持 K-NN 分类
"""
import numpy as np

# dtaidistance 的 C 实现比纯 Python 快 100x+
from dtaidistance import dtw


def dtw_pitch(points_a, points_b):
    """对两条旋律的 pitch 序列做 DTW 对齐

    只取 pitch 维度做 1D DTW，因为：
    1. pitch 是旋律最核心的特征
    2. 1D DTW 比 3D DTW 快得多
    3. 时间维度通过 DTW 的 warping 自动处理了

    Args:
        points_a: (n, 3) 数组 (t, pitch, vel)
        points_b: (m, 3) 数组

    Returns:
        float: DTW 距离
    """
    a = np.ascontiguousarray(points_a[:, 1].ravel().astype(np.float64))
    b = np.ascontiguousarray(points_b[:, 1].ravel().astype(np.float64))
    return dtw.distance(a, b)


def dtw_3d_fast(points_a, points_b):
    """3D DTW，使用 dtaidistance 的窗口加速

    将 (t, pitch, vel) 展开为一维序列，利用 DTW 的对齐能力
    """
    n = min(len(points_a), len(points_b))
    # 如果太长，截断到合理长度（dtaidistance 对长序列较慢）
    a = points_a[:n].ravel().astype(np.float64)
    b = points_b[:n].ravel().astype(np.float64)
    return dtw.distance(a, b, window=int(n * 0.3))
