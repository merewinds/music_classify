"""
K-NN 分类实验：用 Hausdorff 距离 + K 近邻投票分类

配置说明：
- K: 近邻个数（1, 3, 5, 7, 9）
- 距离度量: Hausdorff 距离 H(A,B)
- 投票方式: 简单多数投票（K 个最近邻中哪类最多就选哪类）
- 平局处理: 选 K 个邻居中距离最近的那个的标签
- 训练/测试划分: StratifiedShuffleSplit（分层抽样），80/20
"""
import sys
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import hausdorff
from code.midi_parser import load_adl_data


def knn_predict(test_sample, X_train, y_train, K):
    """对一个测试样本做 K-NN 预测

    1. 算该样本到所有训练样本的 Hausdorff 距离
    2. 取距离最小的 K 个训练样本
    3. K 个邻居投票，多数决定 genre

    平局策略：如果 K 个邻居中多个 genre 票数相同，
    则在这些平局的 genre 中取距离最近的那个邻居的标签
    """
    distances = []
    for j, train_sample in enumerate(X_train):
        d = hausdorff(test_sample, train_sample)
        distances.append((d, y_train[j]))

    # 按距离排序
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:K]

    # 投票
    labels = [label for _, label in k_nearest]
    counter = Counter(labels)
    max_votes = max(counter.values())

    if sum(1 for v in counter.values() if v == max_votes) == 1:
        # 唯一多数
        return counter.most_common(1)[0][0]
    else:
        # 平局：在票数相同的 genre 中，取距离最近的那个
        for d, label in k_nearest:
            if counter[label] == max_votes:
                return label
        return labels[0]


def run_knn_experiment(base_path, genres, K=5, max_per_genre=50, test_size=0.2, random_state=42):
    """配置说明：
        K: 近邻个数
        距离: Hausdorff H(A,B)
        投票: 多数投票，平局时取最近邻
        数据: 最高音旋律线 → 弧长重采样 200 点 → minmax
    """
    print("=" * 60)
    print(f"K-NN 分类实验 (K={K})")
    print("  距离度量: Hausdorff H(A,B)")
    print("  投票方式: 多数投票 (平局取最近邻)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    print(f"  曲风: {genres}")
    print(f"  每种最多 {max_per_genre} 首")
    X, y = load_adl_data(base_path, genres, max_per_genre,
                         normalize_mode='minmax', resample=True, n_samples=200)
    print(f"\n  共加载 {len(X)} 首旋律")

    if len(X) == 0:
        print("  错误：没有加载到任何数据！")
        return None

    # 2. 分层抽样划分 (80/20)
    print("\n[2/3] 划分训练集/测试集...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"  训练集: {len(X_train)} 首")
    print(f"  测试集: {len(X_test)} 首")

    # 3. K-NN 分类
    print(f"\n[3/3] K-NN (K={K}) 分类中...")
    predictions = []
    for i, test_sample in enumerate(X_test):
        pred = knn_predict(test_sample, X_train, y_train, K)
        predictions.append(pred)
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(X_test)}")

    # 4. 评估
    accuracy = accuracy_score(y_test, predictions)
    baseline = 1.0 / len(genres)

    print(f"\n{'=' * 60}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  随机基线: {baseline:.4f} ({baseline*100:.1f}%)")
    print(f"  比基线提升: {(accuracy - baseline)/baseline*100:+.1f}%")
    print(f"{'=' * 60}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions, labels=genres)
    print(f"\n混淆矩阵 (行=真实, 列=预测):")
    header = '  '.join(f'{g[:5]:>5}' for g in genres)
    print(f"        {header}")
    for i, g in enumerate(genres):
        row = '  '.join(f'{cm[i,j]:5d}' for j in range(len(genres)))
        print(f" {g[:5]:>5}  {row}")

    # 分类报告
    print(f"\n分类报告 (K={K}):")
    print(classification_report(y_test, predictions, labels=genres, digits=3))

    return accuracy


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')

    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']

    print("\n" + "#" * 60)
    print("# K-NN 对比实验：不同 K 值的效果")
    print("#" * 60)

    results = {}
    for K in [1, 3, 5, 7, 9]:
        print(f"\n\n{'#' * 60}")
        acc = run_knn_experiment(base, genres, K=K, max_per_genre=80)
        if acc is not None:
            results[K] = acc

    print(f"\n\n{'=' * 60}")
    print("K-NN 对比结果汇总")
    print(f"{'=' * 60}")
    print(f"  曲风: {genres}")
    print(f"  每种上限: 80 首")
    print(f"  特征: 最高音旋律线 → 弧长重采样 200 点 → minmax")
    print(f"  距离: Hausdorff H(A,B)")
    print()
    print(f"  K=1  (1-NN):  {results.get(1, -1):.4f}  ← 之前的实验结果")
    for K in [3, 5, 7, 9]:
        print(f"  K={K}:          {results.get(K, -1):.4f}")
    print(f"  随机基线:     {1/len(genres):.4f}")
    print(f"{'=' * 60}")
