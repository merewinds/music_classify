"""
K-means 聚类实验：基于 Hausdorff 距离矩阵
思路：先用 MDS 把距离矩阵降到特征空间，再 K-means 聚类，看与真实标签的吻合度
"""
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import pairwise_hausdorff_matrix
from code.midi_parser import load_adl_data


def run_kmeans_experiment(base_path, genres, max_per_genre=50, n_clusters=None, random_state=42):
    """基于 Hausdorff 距离矩阵的 K-means 聚类实验

    流程：
    1. 加载数据 -> 提取最高音 -> 重采样 200 点 -> minmax
    2. 计算所有样本两两的 Hausdorff 距离矩阵 (n, n)
    3. MDS 降到特征空间 (n, d)
    4. K-means 聚类
    5. 用真实标签评估：ARI, NMI, 纯度
    """
    print("=" * 60)
    print("K-means 聚类实验 (基于 Hausdorff 距离矩阵)")
    print("=" * 60)

    if n_clusters is None:
        n_clusters = len(genres)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    print(f"  曲风: {genres}")
    print(f"  每种最多 {max_per_genre} 首")
    X, y = load_adl_data(base_path, genres, max_per_genre,
                         normalize_mode='minmax', resample=True, n_samples=200)
    print(f"  共加载 {len(X)} 首旋律")

    if len(X) == 0:
        print("  错误：没有加载到任何数据！")
        return

    # 2. 计算 Hausdorff 距离矩阵
    print(f"\n[2/4] 计算 Hausdorff 距离矩阵 ({len(X)}×{len(X)})...")
    dist_matrix = pairwise_hausdorff_matrix(X)

    # 检查距离矩阵统计
    triu = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    print(f"  距离范围: {triu.min():.4f} ~ {triu.max():.4f}")
    print(f"  距离均值: {triu.mean():.4f} ± {triu.std():.4f}")

    # 3. MDS 降维
    print(f"\n[3/4] MDS 降维 (n_components=10)...")
    mds = MDS(n_components=10, dissimilarity='precomputed',
              random_state=random_state, normalized_stress='auto')
    X_mds = mds.fit_transform(dist_matrix)
    print(f"  MDS 完成: {X_mds.shape}")

    # 4. K-means 聚类
    print(f"\n[4/4] K-means 聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_mds)

    # 统计每个簇的组成
    print(f"\n  各簇组成:")
    label_names = np.array(genres)
    for c in range(n_clusters):
        mask = cluster_labels == c
        indices = np.where(mask)[0]
        actual_labels = [y[i] for i in indices]
        counter = Counter(actual_labels)
        top = counter.most_common(3)
        top_str = ', '.join(f'{l}({n})' for l, n in top)
        print(f"    簇 {c}: {len(indices)} 首 -> {top_str}")

    # 5. 评估
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels, average_method='geometric')
    # Silhouette Score: 在原始距离矩阵上算
    sil = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')

    print(f"\n{'=' * 60}")
    print(f"  聚类评估结果 (k={n_clusters}, 随机基线 ARI ≈ 0)")
    print(f"  Adjusted Rand Index (ARI):  {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"  Silhouette Score:              {sil:.4f}")
    print(f"{'=' * 60}")

    # 也试一下 k=实际类别数 时各指标的含义
    # ARI: 1=完美匹配, 0=随机, 负数=比随机差
    # NMI: 1=完美匹配, 0=随机

    return ari, nmi, sil, cluster_labels, y


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')
    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']
    run_kmeans_experiment(base, genres, max_per_genre=60)
