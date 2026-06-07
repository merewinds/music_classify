"""
DTW (Dynamic Time Warping) 距离实现
"""
import numpy as np


def dtw(A, B):
    """计算两条序列之间的 DTW 距离

    原理：填一个 (n+1)×(m+1) 的累计距离矩阵，
          每个格子 cost[i,j] = d(A[i], B[j]) + min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1])
          返回从 (0,0) 到 (n,m) 的最小累计距离

    Args:
        A: (n, d) 数组，旋律 A 的 n 个点，每行 (t, pitch, vel)
        B: (m, d) 数组，旋律 B 的 m 个点

    Returns:
        float: DTW 距离
    """
    n, m = len(A), len(B)
    # 累计距离矩阵
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # A[i-1] 与 B[j-1] 的欧氏距离
            d = np.sqrt(np.sum((A[i - 1] - B[j - 1]) ** 2))
            # 累计距离 = 当前距离 + 左边/上边/左上角的最小值
            cost[i, j] = d + min(cost[i - 1, j],     # 上方 → A 多对一
                                 cost[i, j - 1],     # 左边 → B 多对一
                                 cost[i - 1, j - 1]) # 左上 → 一对一

    return cost[n, m]


def dtw_fast(A, B):
    """优化版 DTW：只用两行数组，节省内存"""
    n, m = len(A), len(B)
    # 只用两行
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0

    for i in range(1, n + 1):
        curr[0] = np.inf
        for j in range(1, m + 1):
            d = np.sqrt(np.sum((A[i - 1] - B[j - 1]) ** 2))
            curr[j] = d + min(prev[j],       # 上方
                              curr[j - 1],   # 左边
                              prev[j - 1])   # 左上角
        prev, curr = curr, prev

    return prev[m]
