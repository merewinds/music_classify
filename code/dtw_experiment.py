"""
DTW vs Hausdorff 对比实验：用两种距离做 K-NN (K=5) 分类
"""
import sys, os, time
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.midi_parser import load_adl_data
from code.hausdorff import hausdorff
from code.dtw_distance import dtw_fast as dtw
from code.frechet_distance import frechet


def knn_predict(test_sample, X_train, y_train, K, distance_fn):
    """通用 K-NN 预测，接受不同的距离函数"""
    distances = [(distance_fn(test_sample, train_sample), label)
                 for train_sample, label in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])

    k_nearest = distances[:K]
    labels = [label for _, label in k_nearest]
    counter = Counter(labels)
    max_votes = max(counter.values())

    if sum(1 for v in counter.values() if v == max_votes) == 1:
        return counter.most_common(1)[0][0]
    else:
        for _, label in k_nearest:
            if counter[label] == max_votes:
                return label
        return k_nearest[0][1]


def run_experiment(base_path, genres, max_per_genre=60, K=5):
    print("=" * 60)
    print("DTW vs Hausdorff 对比实验")
    print(f"  数据: {genres} (每种最多{max_per_genre}首)")
    print(f"  K-NN: K={K}")
    print("=" * 60)

    # 1. 加载
    print("\n[1/2] 加载数据...")
    X, y = load_adl_data(base_path, genres, max_per_genre,
                         normalize_mode='minmax', resample=True, n_samples=200)
    print(f"  共加载 {len(X)} 首旋律")

    if len(X) == 0:
        return

    # 2. 划分 80/20
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    print(f"  训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 3. 对比两种距离
    results = {}
    for name, dist_fn in [("Hausdorff", hausdorff), ("DTW", dtw), ("Frechet", frechet)]:
        print(f"\n[2/2] {name} K-NN (K={K})...")
        t0 = time.time()

        predictions = []
        for i, test_pt in enumerate(X_test):
            pred = knn_predict(test_pt, X_train, y_train, K, dist_fn)
            predictions.append(pred)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {name} 进度: {i+1}/{len(X_test)} ({elapsed:.0f}s)")

        acc = accuracy_score(y_test, predictions)
        elapsed = time.time() - t0
        results[name] = acc

        print(f"\n  {name} K-NN (K={K}):")
        print(f"    准确率: {acc:.4f} ({acc*100:.1f}%)")
        print(f"    用时:   {elapsed:.0f}s")

        # 混淆矩阵
        cm = confusion_matrix(y_test, predictions, labels=genres)
        print(f"  混淆矩阵:")
        header = '  '.join(f'{g[:4]:>4}' for g in genres)
        print(f"         {header}")
        for i, g in enumerate(genres):
            row = '  '.join(f'{cm[i,j]:4d}' for j in range(len(genres)))
            print(f"  {g[:4]:>4}  {row}")

    # 汇总
    baseline = 1.0 / len(genres)
    print(f"\n{'=' * 60}")
    print(f"汇总对比")
    print(f"{'=' * 60}")
    print(f"  曲风: {genres} ({len(genres)} 类)")
    print(f"  特征: 最高音旋律线 → 重采样 200 点 → minmax")
    print(f"  分类: K-NN K={K}")
    print()
    for name, acc in results.items():
        impr = (acc - baseline) / baseline * 100
        print(f"  {name:10s}:  {acc:.4f} ({acc*100:.1f}%)  [比基线+{impr:.0f}%]")
    print(f"  随机基线:   {baseline:.4f} ({baseline*100:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')
    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']
    run_experiment(base, genres, max_per_genre=20, K=5)
