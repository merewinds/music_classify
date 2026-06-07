"""
离散 Fréchet 距离实现

离散 Fréchet 距离和 DTW 很像，区别在于：
  DTW:      走完整条路径的 累计距离最小化（求和）
  Fréchet:  走完整条路径的 最大单步距离最小化（取 max）
"""
import numpy as np


def frechet(A, B):
    """离散 Fréchet 距离

    动态规划填表：
      ca[i,j] = max( d(A[i], B[j]), min(ca[i-1,j], ca[i,j-1], ca[i-1,j-1]) )

    Returns:
        float: Fréchet 距离
    """
    n, m = len(A), len(B)
    # ca[i,j] = 走到 A[i] 和 B[j] 时的最小"最大单步距离"
    ca = np.full((n, m), np.inf)

    for i in range(n):
        for j in range(m):
            d = np.sqrt(np.sum((A[i] - B[j]) ** 2))
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), d)

    return ca[n - 1, m - 1]
