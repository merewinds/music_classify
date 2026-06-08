"""
max-Hausdorff vs MHD（Modified Hausdorff Distance）对比实验
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import hausdorff, mhd
from code.midi_parser import load_adl_data


def knn_cv(X, y, genres, K, dist_func, n_splits=5):
    """K-NN 交叉验证"""
    y_num = np.array([genres.index(yi) for yi in y])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, all_cms = [], []

    for train_idx, test_idx in skf.split(X, y_num):
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]
        preds = []

        for ts in X_te:
            dists = [dist_func(ts, tr) for tr in X_tr]
            top_k = np.argsort(dists)[:K]
            k_labels = [y_tr[i] for i in top_k]
            counter = Counter(k_labels)
            max_votes = max(counter.values())
            if sum(1 for v in counter.values() if v == max_votes) == 1:
                preds.append(counter.most_common(1)[0][0])
            else:
                preds.append(k_labels[0])

        accs.append(accuracy_score(y_te, preds))
        all_cms.append(confusion_matrix(y_te, preds, labels=genres))

    return np.mean(accs), np.std(accs), np.mean(all_cms, axis=0)


def knn_single(X, y, genres, K, dist_func, test_size=0.2):
    """K-NN 单次 80/20 划分"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    y_num = np.array([genres.index(yi) for yi in y])
    train_idx, test_idx = next(sss.split(X, y_num))
    X_tr = [X[i] for i in train_idx]
    y_tr = [y[i] for i in train_idx]
    X_te = [X[i] for i in test_idx]
    y_te = [y[i] for i in test_idx]
    preds = []

    for ts in X_te:
        dists = [dist_func(ts, tr) for tr in X_tr]
        top_k = np.argsort(dists)[:K]
        k_labels = [y_tr[i] for i in top_k]
        counter = Counter(k_labels)
        max_votes = max(counter.values())
        if sum(1 for v in counter.values() if v == max_votes) == 1:
            preds.append(counter.most_common(1)[0][0])
        else:
            preds.append(k_labels[0])

    return accuracy_score(y_te, preds), y_te, preds


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')
    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']

    print("#" * 60)
    print("# max-Hausdorff vs MHD 对比实验")
    print("#" * 60)

    print("\n[1/3] 加载数据...")
    X, y = load_adl_data(base, genres, 120,
                         normalize_mode='minmax', resample=True, n_samples=200)
    print(f"  共 {len(X)} 首旋律\n")

    # 5-fold CV
    print("[2/3] 5-fold 交叉验证 (K=5)...")
    for K in [3, 5, 7]:
        hd_mean, hd_std, hd_cm = knn_cv(X, y, genres, K, hausdorff)
        mhd_mean, mhd_std, mhd_cm = knn_cv(X, y, genres, K, mhd)
        print(f"\n  K={K}:")
        print(f"    Hausdorff (max): {hd_mean:.4f} +/- {hd_std:.4f}")
        print(f"    MHD      (mean): {mhd_mean:.4f} +/- {mhd_std:.4f}")
        print(f"    差异: {mhd_mean - hd_mean:+.4f}")

    # K=5 详细混淆矩阵对比
    print(f"\n\n[3/3] K=5 详细对比...")
    _, _, hd_cm = knn_cv(X, y, genres, 5, hausdorff)
    _, _, mhd_cm = knn_cv(X, y, genres, 5, mhd)

    print(f"\n  Hausdorff (max) 平均混淆矩阵 (%):")
    header = '  '.join(f'{g[:5]:>5}' for g in genres)
    print(f"        {header}")
    for i, g in enumerate(genres):
        total = hd_cm[i].sum()
        pct = hd_cm[i] / total * 100 if total else hd_cm[i]
        row = '  '.join(f'{p:>4.0f}%' for p in pct)
        print(f" {g[:5]:>5}  {row}")

    print(f"\n  MHD (mean) 平均混淆矩阵 (%):")
    print(f"        {header}")
    for i, g in enumerate(genres):
        total = mhd_cm[i].sum()
        pct = mhd_cm[i] / total * 100 if total else mhd_cm[i]
        row = '  '.join(f'{p:>4.0f}%' for p in pct)
        print(f" {g[:5]:>5}  {row}")

    # K=5 单次划分 vs DTW
    print(f"\n\n  与 DTW 对标 (K=5, 80/20 单次):")
    hd_acc, _, _ = knn_single(X, y, genres, 5, hausdorff)
    mhd_acc, _, _ = knn_single(X, y, genres, 5, mhd)
    print(f"    Hausdorff (max): {hd_acc:.4f}")
    print(f"    MHD      (mean): {mhd_acc:.4f}")
    print(f"    DTW (ref):        0.397 (之前结果)")

    print(f"\n{'#' * 60}")
    print("Done!")
