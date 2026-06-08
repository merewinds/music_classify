"""
Hausdorff 有效性综合分析
补全整个论证链条中的所有缺漏环节

运行: python -W ignore code/hausdorff_analysis.py 2>/dev/null
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import MDS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import hausdorff, pairwise_hausdorff_matrix
from code.midi_parser import load_adl_data

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _minmax(p):
    norm = np.zeros_like(p)
    for i in range(3):
        c = p[:, i]
        if c.max() > c.min():
            norm[:, i] = (c - c.min()) / (c.max() - c.min())
        else:
            norm[:, i] = 0.5
    return norm


# ═══════════════════════════════════════════════════════════════
# E1: 合成数据验证 - 验证 Hausdorff 距离本身的行为是否合理
# ═══════════════════════════════════════════════════════════════
def run_synthetic_test():
    print("\n" + "=" * 60)
    print("E1: 合成数据验证")
    print("=" * 60)

    # 构造 5 条测试旋律
    # 1) 音阶上行: C D E F G A B C
    up = np.array([[i, 60 + i, 80] for i in range(8)], dtype=np.float64)
    # 2) 音阶下行: C B A G F E D C
    down = np.array([[i, 60 + (7 - i), 80] for i in range(8)], dtype=np.float64)
    # 3) 琶音: C E G C E G C C (大跳)
    arp = np.array([[i, 60 + [0, 4, 7, 12, 4, 7, 12, 0][i], 80]
                    for i in range(8)], dtype=np.float64)
    # 4) 音阶+噪声 (小扰动)
    noisy = up.copy()
    noisy[:, 1] += np.random.normal(0, 0.5, 8)
    noisy[:, 2] += np.random.normal(0, 2, 8)

    # 5) 随机游走
    rw = np.zeros((20, 3), dtype=np.float64)
    rw[0] = [0, 60, 80]
    for i in range(1, 20):
        rw[i] = rw[i-1] + [np.random.uniform(0.3, 0.8), np.random.randint(-3, 4), np.random.randint(-10, 11)]
    rw[:, 2] = np.clip(rw[:, 2], 0, 127)

    # 归一化
    samples = [
        ("向上音阶", _minmax(up)),
        ("向下音阶", _minmax(down)),
        ("琶音", _minmax(arp)),
        ("音阶+噪", _minmax(noisy)),
        ("随机游走", _minmax(rw)),
    ]

    print("\n  Hausdorff 距离矩阵:")
    print(f"  {'':12s}", end='')
    for name, _ in samples:
        print(f"  {name:>8s}", end='')
    print()

    for name_a, a in samples:
        print(f"  {name_a:12s}", end='')
        for name_b, b in samples:
            d = hausdorff(a, b)
            print(f"  {d:>8.4f}", end='')
        print()

    # 验证:
    d_self = hausdorff(samples[0][1], samples[0][1])
    d_noise = hausdorff(samples[0][1], samples[3][1])
    d_diff = hausdorff(samples[0][1], samples[1][1])
    d_random = hausdorff(samples[0][1], samples[4][1])

    checks = [
        ("自身距离应为0", d_self < 1e-6, f"{d_self:.6f}"),
        ("加噪应很小(<0.1)", d_noise < 0.1, f"{d_noise:.4f}"),
        ("噪声<下行", d_noise < d_diff, f"{d_noise:.4f} < {d_diff:.4f} = {d_noise < d_diff}"),
        ("下行<随机", d_diff < d_random, f"{d_diff:.4f} < {d_random:.4f} = {d_diff < d_random}"),
    ]

    print(f"\n  Hausdorff 距离验证:")
    all_pass = True
    for desc, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {desc} ({detail})")
    print(f"  总体: {'PASS' if all_pass else '部分FAIL'}")
    return all_pass


# ═══════════════════════════════════════════════════════════════
# E2: 同类 vs 异类距离分布 + 统计检验
# ═══════════════════════════════════════════════════════════════
def run_distance_distribution_test(X, y, genres):
    print("\n" + "=" * 60)
    print("E2: 同类 vs 异类 Hausdorff 距离分布")
    print("=" * 60)

    n = len(X)
    within, between = [], []

    for i in range(n):
        for j in range(i+1, n):
            d = hausdorff(X[i], X[j])
            if y[i] == y[j]:
                within.append(d)
            else:
                between.append(d)

    within = np.array(within)
    between = np.array(between)

    print(f"\n  同类距离 ({len(within)} 对):")
    print(f"    均值={within.mean():.4f}  中位数={np.median(within):.4f}  std={within.std():.4f}")
    print(f"  异类距离 ({len(between)} 对):")
    print(f"    均值={between.mean():.4f}  中位数={np.median(between):.4f}  std={between.std():.4f}")

    # Mann-Whitney U 检验
    u_stat, p_value = stats.mannwhitneyu(within, between, alternative='less')
    print(f"\n  Mann-Whitney U 检验 (H0: 同类 >= 异类):")
    print(f"    U统计量={u_stat:.0f}  p值={p_value:.6f}")
    print(f"    结论: 同类距离{'显著' if p_value < 0.05 else '不显著'}小于异类距离 (p={'<' if p_value < 0.05 else '>='}0.05)")

    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([within, between], labels=['Within-Genre', 'Between-Genre'])
    ax.set_ylabel('Hausdorff Distance')
    ax.set_title('Within-Genre vs Between-Genre Hausdorff Distance')
    ax.grid(alpha=0.3)
    out_path = os.path.join(OUTPUT_DIR, 'within_vs_between.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  箱线图已保存: {out_path}")

    # 各 genre 对的平均距离矩阵
    print(f"\n  平均距离矩阵 (行/列 = genre):")
    header = '  '.join(f'{g[:6]:>6}' for g in genres)
    print(f"        {header}")
    for i, g1 in enumerate(genres):
        row_vals = []
        for g2 in genres:
            ds = []
            for ii in range(n):
                if y[ii] != g1: continue
                for jj in range(n):
                    if y[jj] != g2 or ii >= jj: continue
                    ds.append(hausdorff(X[ii], X[jj]))
            row_vals.append(np.mean(ds) if ds else 0)
        row_str = '  '.join(f'{v:>6.3f}' for v in row_vals)
        print(f" {g1[:6]:>6}  {row_str}")

    return within, between


# ═══════════════════════════════════════════════════════════════
# E3: MDS 可视化
# ═══════════════════════════════════════════════════════════════
def run_mds_visualization(X, y, genres):
    print("\n" + "=" * 60)
    print("E3: MDS 可视化")
    print("=" * 60)

    n = len(X)
    print(f"  计算 {n}x{n} Hausdorff 距离矩阵...")
    dist_matrix = pairwise_hausdorff_matrix(X, verbose=False)

    print(f"  MDS 降维到 2D...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    X_mds = mds.fit_transform(dist_matrix)

    colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
    color_map = dict(zip(genres, colors))

    fig, ax = plt.subplots(figsize=(10, 8))
    for g in genres:
        mask = np.array(y) == g
        ax.scatter(X_mds[mask, 0], X_mds[mask, 1],
                   c=[color_map[g]], label=g, alpha=0.7, s=30)

    ax.set_title("MDS: Hausdorff Distance Space (5 Genres)")
    ax.legend(fontsize=10)
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.grid(alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, 'mds_hausdorff.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  图片已保存: {out_path}")
    return X_mds


# ═══════════════════════════════════════════════════════════════
# E4: K-NN 交叉验证
# ═══════════════════════════════════════════════════════════════
def run_cross_validation(X, y, genres, K=5, n_splits=5):
    print("\n" + "=" * 60)
    print(f"E4: K-NN (K={K}) {n_splits}-fold 交叉验证")
    print("=" * 60)

    y_num = np.array([genres.index(yi) for yi in y])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []
    all_cms = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_num)):
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]

        preds = []
        for ts in X_te:
            dists = [hausdorff(ts, tr) for tr in X_tr]
            top_k = np.argsort(dists)[:K]
            k_labels = [y_tr[i] for i in top_k]
            counter = Counter(k_labels)
            max_votes = max(counter.values())
            if sum(1 for v in counter.values() if v == max_votes) == 1:
                preds.append(counter.most_common(1)[0][0])
            else:
                preds.append(k_labels[0])

        acc = accuracy_score(y_te, preds)
        fold_accs.append(acc)
        all_cms.append(confusion_matrix(y_te, preds, labels=genres))
        print(f"  Fold {fold+1}: {acc:.4f} ({acc*100:.1f}%)")

    acc_mean = np.mean(fold_accs)
    acc_std = np.std(fold_accs)
    baseline = 1 / len(genres)

    print(f"\n  {n_splits}-fold CV 结果:")
    print(f"    准确率: {acc_mean:.4f} +/- {acc_std:.4f}")
    print(f"    范围: [{acc_mean-acc_std:.4f}, {acc_mean+acc_std:.4f}]")
    print(f"    随机基线: {baseline:.4f}")

    # 平均混淆矩阵
    avg_cm = np.mean(all_cms, axis=0)
    print(f"\n  平均混淆矩阵 (归一化 %):")
    header = '  '.join(f'{g[:5]:>5}' for g in genres)
    print(f"        {header}")
    for i, g in enumerate(genres):
        total = avg_cm[i].sum()
        pct = avg_cm[i] / total * 100 if total > 0 else avg_cm[i]
        row = '  '.join(f'{p:>4.0f}%' for p in pct)
        print(f" {g[:5]:>5}  {row}")

    # 画图
    fig, ax = plt.subplots(figsize=(8, 6))
    folds = np.arange(1, n_splits + 1)
    ax.bar(folds, fold_accs, width=0.5)
    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.2f})')
    ax.axhline(y=acc_mean, color='g', linestyle='-', label=f'Mean ({acc_mean:.3f})')
    ax.fill_between(folds, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color='g')
    ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
    ax.set_title(f'K-NN (K={K}) {n_splits}-fold Cross-Validation')
    ax.legend(); ax.grid(alpha=0.3)
    out_path = os.path.join(OUTPUT_DIR, 'cv_knn.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  图片已保存: {out_path}")

    return fold_accs


# ═══════════════════════════════════════════════════════════════
# E5: 类别数敏感性
# ═══════════════════════════════════════════════════════════════
def run_genre_scale_analysis(base_path, genre_sets, K=5, max_per_genre=60):
    print("\n" + "=" * 60)
    print("E5: 类别数敏感性分析")
    print("=" * 60)

    results = []
    for name, gs in genre_sets:
        print(f"\n  加载 {name} ({len(gs)} genres)...")
        X, y = load_adl_data(base_path, gs, max_per_genre,
                             normalize_mode='minmax', resample=True, n_samples=200)
        if len(X) == 0:
            continue

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        y_num = np.array([gs.index(yi) for yi in y])
        train_idx, test_idx = next(sss.split(X, y_num))
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]

        preds = []
        for ts in X_te:
            dists = [hausdorff(ts, tr) for tr in X_tr]
            top_k = np.argsort(dists)[:K]
            k_labels = [y_tr[i] for i in top_k]
            counter = Counter(k_labels)
            max_votes = max(counter.values())
            if sum(1 for v in counter.values() if v == max_votes) == 1:
                preds.append(counter.most_common(1)[0][0])
            else:
                preds.append(k_labels[0])

        acc = accuracy_score(y_te, preds)
        baseline = 1 / len(gs)
        results.append((len(gs), acc, baseline, name))

    print(f"\n  类别数敏感性汇总:")
    print(f"  {'N':>4s}  {'Accuracy':>9s}  {'Baseline':>9s}  {'Improve':>8s}  {'Genres'}")
    for n_gen, acc, bl, nm in results:
        impr = (acc - bl) / bl * 100
        print(f"  {n_gen:>4d}  {acc:>9.4f}  {bl:>9.4f}  {impr:>+7.0f}%  {nm}")

    # 画图
    if results:
        fig, ax = plt.subplots(figsize=(8, 6))
        ns = [r[0] for r in results]
        accs = [r[1] for r in results]
        bls = [r[2] for r in results]
        ax.plot(ns, accs, 'o-', label='K-NN (K=5)')
        ax.plot(ns, bls, 's--', label='Random Baseline')
        ax.set_xlabel('Number of Genres')
        ax.set_ylabel('Accuracy')
        ax.set_title('K-NN Accuracy vs Number of Genres')
        ax.legend()
        ax.grid(alpha=0.3)
        out_path = os.path.join(OUTPUT_DIR, 'genre_scale.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  图片已保存: {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')

    print("#" * 60)
    print("# Hausdorff Distance Genre Classification Analysis")
    print("#" * 60)

    # E1: Synthetic data validation
    run_synthetic_test()

    # E2-E4: Real data (5 genres, 60 each)
    genres5 = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']
    print("\n\nLoading real data (5 genres, 120 each)...")
    X, y = load_adl_data(base, genres5, 120,
                         normalize_mode='minmax', resample=True, n_samples=200)

    # E2: Distance distribution
    run_distance_distribution_test(X, y, genres5)

    # E3: MDS visualization
    run_mds_visualization(X, y, genres5)

    # E4: Cross-validation
    run_cross_validation(X, y, genres5, K=5, n_splits=5)

    # E5: Genre scale analysis
    genre_sets = [
        ("2", ['Classical', 'Jazz']),
        ("+Rock", ['Classical', 'Jazz', 'Rock']),
        ("+Blues", ['Classical', 'Jazz', 'Rock', 'Blues']),
        ("+Electronic", ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']),
        ("+Pop", ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic', 'Pop']),
        ("+Country", ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic', 'Pop', 'Country']),
    ]
    run_genre_scale_analysis(base, genre_sets, K=5, max_per_genre=120)

    print(f"\n\n{'#' * 60}")
    print("Analysis complete!")
    print(f"  Figures saved to: {OUTPUT_DIR}")
    print(f"{'#' * 60}")
