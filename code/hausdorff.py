"""
Hausdorff 距离核心算法
"""
import numpy as np
from scipy.spatial import KDTree


def directed_hausdorff(A, B):
    """有向 Hausdorff 距离 h(A,B) = max_{a∈A} min_{b∈B} ||a-b||"""
    tree = KDTree(B)
    distances, _ = tree.query(A)
    return float(np.max(distances))


def hausdorff(A, B):
    """双向 Hausdorff 距离 H(A,B) = max(h(A,B), h(B,A))"""
    return max(directed_hausdorff(A, B), directed_hausdorff(B, A))


# ─── Modified Hausdorff Distance (MHD) ────────────────────────
# 针对原版 Hausdorff 使用 max 对离群点敏感的问题，
# 将 max 替换为 mean，使距离更平滑、更关注整体分布

def directed_mhd(A, B):
    """有向 MHD: h_mhd(A,B) = mean_{a∈A} min_{b∈B} ||a-b||"""
    tree = KDTree(B)
    distances, _ = tree.query(A)
    return float(np.mean(distances))


def mhd(A, B):
    """双向 MHD: H_mhd(A,B) = max(mean(A→B), mean(B→A))

    注意：双向仍保留 max，因为两个方向的不对称信息仍然有意义。
    取 max 表示"两个方向中平均偏差较大的那个方向"。
    """
    return max(directed_mhd(A, B), directed_mhd(B, A))


def pairwise_hausdorff_matrix(datasets, verbose=True):
    """计算数据集两两之间的 Hausdorff 距离矩阵

    Args:
        datasets: list of (N_i, 3) arrays
        verbose: 是否打印进度

    Returns:
        (n, n) 对称距离矩阵
    """
    n = len(datasets)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hausdorff(datasets[i], datasets[j])
            matrix[i, j] = d
            matrix[j, i] = d
        if verbose and (i + 1) % 20 == 0:
            print(f"  Hausdorff 矩阵进度: {i+1}/{n}")
    return matrix
