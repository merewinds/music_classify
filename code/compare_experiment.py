"""
Hausdorff vs DTW 对比实验：K-NN 分类

对比两种距离度量在相同数据下的 K-NN 分类效果：
- Hausdorff: 无序点集匹配，3D (t, pitch, vel)
- DTW: 有序时序对齐，1D pitch 序列（dtaidistance C 加速）
"""
import sys
import os
import time
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import hausdorff
from code.dtw_distance import dtw_pitch
from code.midi_parser import load_adl_data


def knn_predict_dist(test_sample, X_train, y_train, K, dist_func):
    distances = []
    for train_sample in X_train:
        d = dist_func(test_sample, train_sample)
        distances.append(d)

    indices = np.argsort(distances)[:K]
    k_labels = [y_train[i] for i in indices]

    counter = Counter(k_labels)
    max_votes = max(counter.values())

    if sum(1 for v in counter.values() if v == max_votes) == 1:
        return counter.most_common(1)[0][0]
    else:
        return k_labels[0]


def run_comparison(base_path, genres, K=5, max_per_genre=60, test_size=0.2, random_state=42):
    """对比 Hausdorff vs DTW 的效果

    数据处理流程（两者共用）：
      最高音旋律线 → 弧长重采样 200 点 → minmax

    差异：
      Hausdorff → 3D (t, pitch, vel) 无序匹配
      DTW      → pitch 序列 1D 时序对齐

    K=5 是将要测试的 K 值
    """
    print("=" * 70)
    print("Hausdorff vs DTW 对比实验")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    print(f"  曲风: {genres}")
    print(f"  每种上限: {max_per_genre} 首")
    X, y = load_adl_data(base_path, genres, max_per_genre,
                         normalize_mode='minmax', resample=True, n_samples=200)
    print(f"\n  共加载 {len(X)} 首旋律")

    if len(X) == 0:
        return

    # 2. 拆分
    print("\n[2/4] 划分训练/测试集 (80/20)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    print(f"  训练集: {len(X_train)} 首")
    print(f"  测试集: {len(X_test)} 首")

    # 3. 定义距离函数
    methods = []

    # Hausdorff: 3D 无序
    methods.append(("Hausdorff (3D)", lambda a, b: hausdorff(a, b)))

    # DTW: 1D pitch 时序对齐（dtaidistance C 加速）
    methods.append(("DTW (pitch seq)", lambda a, b: dtw_pitch(a, b)))

    results = {}

    for method_name, dist_func in methods:
        print(f"\n[3/4] {method_name}...")
        start = time.time()
        predictions = []
        for i, ts in enumerate(X_test):
            pred = knn_predict_dist(ts, X_train, y_train, K, dist_func)
            predictions.append(pred)
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(X_test)}")
        elapsed = time.time() - start
        acc = accuracy_score(y_test, predictions)
        results[method_name] = {'accuracy': acc, 'predictions': predictions, 'time': elapsed}
        print(f"  准确率: {acc:.4f} ({acc*100:.1f}%)  耗时: {elapsed:.0f}s")

    # 4. 汇总
    baseline = 1.0 / len(genres)
    print(f"\n{'=' * 70}")
    print("对比结果汇总")
    print(f"{'=' * 70}")
    print(f"  曲风: {genres}")
    print(f"  每种上限: {max_per_genre} 首")
    print(f"  K-NN K={K}")
    print(f"  随机基线: {baseline:.4f} ({baseline*100:.1f}%)")
    print()
    for name, r in results.items():
        acc = r['accuracy']
        print(f"  {name:25s}: {acc:.4f} ({acc*100:.1f}%)  "
              f"比基线 +{(acc-baseline)/baseline*100:.0f}%  "
              f"耗时 {r['time']:.0f}s")

    # DTW 详细报告
    if "DTW (pitch seq)" in results:
        print(f"\n--- DTW (pitch seq) 详细报告 (K={K}) ---")
        y_pred = results["DTW (pitch seq)"]['predictions']
        print(classification_report(y_test, y_pred, labels=genres, digits=3))
        cm = confusion_matrix(y_test, y_pred, labels=genres)
        print("混淆矩阵 (行=真实, 列=预测):")
        header = '  '.join(f'{g[:5]:>5}' for g in genres)
        print(f"        {header}")
        for i, g in enumerate(genres):
            row = '  '.join(f'{cm[i,j]:5d}' for j in range(len(genres)))
            print(f" {g[:5]:>5}  {row}")

    return results


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')
    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']
    run_comparison(base, genres, K=5, max_per_genre=60)
