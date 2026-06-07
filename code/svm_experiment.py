"""
SVM 分类实验：基于 Hausdorff 距离矩阵 + RBF 核
"""
import sys, os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import hausdorff, pairwise_hausdorff_matrix
from code.midi_parser import load_adl_data


def run_svm_experiment(base_path, genres, max_per_genre=80, random_state=42):
    print("=" * 60)
    print("SVM 分类实验 (预计算核: Hausdorff + RBF)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    X, y = load_adl_data(base_path, genres, max_per_genre,
                         normalize_mode='minmax', resample=True, n_samples=200)
    print(f"  共加载 {len(X)} 首旋律")

    if len(X) == 0:
        return None

    y = np.array(y)
    n_total = len(X)

    # 2. 分层 80/20 划分
    print("\n[2/4] 划分训练集/测试集...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))

    X_train = [X[i] for i in train_idx]
    y_train = y[train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = y[test_idx]

    print(f"  训练集: {len(X_train)} 首 | 测试集: {len(X_test)} 首")

    # 3. 训练集 Hausdorff 距离矩阵 → RBF 核矩阵
    print("\n[3/4] 计算训练集核矩阵...")
    D_train = pairwise_hausdorff_matrix(X_train)
    # gamma = 1 / 距离方差
    triu = D_train[np.triu_indices_from(D_train, k=1)]
    gamma_val = 1.0 / (triu.var() + 1e-10)
    print(f"  gamma = {gamma_val:.4f}")
    K_train = np.exp(-gamma_val * D_train ** 2)

    # 4. 训练 + 测试不同 C
    print("\n[4/4] 训练 & 预测...")

    results = {}
    for C in [0.1, 1, 10, 100]:
        svm = SVC(kernel='precomputed', C=C, random_state=random_state)
        svm.fit(K_train, y_train)

        # 测试：计算双向 Hausdorff H(测试曲, 训练曲)
        D_test = np.zeros((len(X_test), len(X_train)))
        for i, test_pt in enumerate(X_test):
            for j, train_pt in enumerate(X_train):
                D_test[i, j] = hausdorff(test_pt, train_pt)
            if (i + 1) % 20 == 0:
                print(f"  计算测试距离 ({i+1}/{len(X_test)})")

        K_test = np.exp(-gamma_val * D_test ** 2)
        pred = svm.predict(K_test)
        acc = accuracy_score(y_test, pred)
        results[C] = acc
        print(f"  C={C:4}: 准确率 {acc:.4f} ({acc*100:.1f}%)")

    best_C = max(results, key=results.get)
    print(f"\n{'=' * 60}")
    print(f"SVM 结果汇总 (Hausdorff + RBF)")
    print(f"{'=' * 60}")
    print(f"  曲风: {genres} ({len(genres)} 类)")
    print(f"  训练/测试: {len(X_train)}/{len(X_test)}")
    print(f"  gamma: {gamma_val:.4f}")
    for C, acc in results.items():
        tag = " <-- 最佳" if acc == max(results.values()) else ""
        print(f"  C={C:6.1f}:  {acc:.4f} ({acc*100:.1f}%){tag}")
    print(f"  随机基线:  {1/len(genres):.4f} ({(1/len(genres))*100:.1f}%)")
    print(f"  K-NN K=5:  0.346 (34.6%) [之前实验]")

    return results


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')
    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']
    run_svm_experiment(base, genres, max_per_genre=60)
