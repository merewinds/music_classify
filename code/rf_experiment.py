"""
基于重采样曲线的特征工程 + 随机森林
数据流：重采样 200 点 → 只归一化时间到 [0,1]，保留原始 pitch/vel → 提取特征
与 Hausdorff/MHD 在重采样之后分叉，对比公平
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.hausdorff import hausdorff
from code.midi_parser import load_adl_data


def extract_curve_features(curve):
    """从重采样曲线 (200×3) 提取特征
    curve 的 pitch/vel 是原始值（0-127），t 已归一化到 [0,1]
    """
    f = {}
    t = curve[:, 0]; p = curve[:, 1]; v = curve[:, 2]
    n = len(curve)

    # ─── A. 统计特征（依赖原始值） ────────────────
    f['pitch_mean'] = np.mean(p)
    f['pitch_std'] = np.std(p)
    f['pitch_range'] = np.max(p) - np.min(p)
    f['vel_mean'] = np.mean(v)
    f['vel_std'] = np.std(v)
    f['vel_range'] = np.max(v) - np.min(v)

    # pitch class 直方图（原始半音值 % 12，真实音乐意义）
    pc = (p.astype(int) % 12)
    hist = np.zeros(12)
    for x in pc: hist[x] += 1
    hist = hist / (hist.sum() + 1e-10)
    for i in range(12): f[f'pc_{i}'] = hist[i]

    # 音高熵（反映音高多样性）
    bins = np.linspace(0, 127, 24)
    digitized = np.digitize(p, bins)
    _, counts = np.unique(digitized, return_counts=True)
    probs = counts / max(counts.sum(), 1)
    f['pitch_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))

    # ─── B. 时序差分特征 ─────────────────────────
    dp = np.diff(p)
    dv = np.diff(v)

    f['dp_mean'] = np.mean(np.abs(dp))
    f['dp_std'] = np.std(dp)
    f['dp_max'] = np.max(np.abs(dp))
    f['dp_positive'] = np.sum(dp > 0) / max(len(dp), 1)
    f['dp_negative'] = np.sum(dp < 0) / max(len(dp), 1)
    f['dp_zero'] = np.sum(dp == 0) / max(len(dp), 1)
    f['dv_mean'] = np.mean(np.abs(dv))
    f['dv_range'] = np.max(dv) - np.min(dv)
    # 大跳比例（真实半音值）
    f['large_leap'] = np.sum(np.abs(dp) >= 5) / max(len(dp), 1)

    # ─── C. 方向 N-gram ──────────────────────────
    dirs = np.sign(dp)
    patterns = {}
    for i in range(len(dirs) - 2):
        pat = (dirs[i], dirs[i+1], dirs[i+2])
        patterns[pat] = patterns.get(pat, 0) + 1
    total = sum(patterns.values()) + 1e-10
    for pat_str in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                    (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1),
                    (0,0,0), (1,0,0), (0,0,1), (-1,0,0)]:
        f[f'ngram_{pat_str[0]}{pat_str[1]}{pat_str[2]}'] = patterns.get(pat_str, 0) / total

    # ─── D. 自相关（旋律重复性） ──────────────────
    pn = (p - np.mean(p)) / (np.std(p) + 1e-10)
    ac = np.correlate(pn, pn, mode='full')
    ac = ac[len(ac)//2:] / max(ac[len(ac)//2], 1e-10)
    f['ac_lag1'] = ac[1] if len(ac) > 1 else 0
    f['ac_lag2'] = ac[2] if len(ac) > 2 else 0
    f['ac_lag5'] = ac[5] if len(ac) > 5 else 0

    # ─── E. 曲线形状特征 ─────────────────────────
    diffs = np.diff(curve[:, :2], axis=0)  # 只用 t 和 pitch 维度
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    f['arclength'] = np.sum(seg_len)
    end_dist = np.sqrt(np.sum((curve[-1, :2] - curve[0, :2]) ** 2))
    f['curvature_ratio'] = f['arclength'] / max(end_dist, 1e-10)

    seg = n // 3
    p1 = np.mean(p[:seg]); p2 = np.mean(p[seg:2*seg]); p3 = np.mean(p[2*seg:])
    f['arch_rise'] = p2 - p1
    f['arch_fall'] = p2 - p3
    f['arch_symmetry'] = abs((p2 - p1) - (p2 - p3))

    return f


FEATURE_NAMES = None


def curve_to_vec(curve):
    global FEATURE_NAMES
    feats = extract_curve_features(curve)
    if FEATURE_NAMES is None:
        FEATURE_NAMES = list(feats.keys())
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float64)


def load_rf_data(base_path, genres, max_per_genre=120):
    """加载数据：重采样 200 点 + time_only 归一化（保留 pitch/vel）"""
    points_list, labels = load_adl_data(base_path, genres, max_per_genre,
                                         normalize_mode='time_only',
                                         resample=True, n_samples=200)
    X = np.array([curve_to_vec(p) for p in points_list])
    return X, np.array(labels)


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'adl-piano-midi')
    genres = ['Classical', 'Jazz', 'Rock', 'Blues', 'Electronic']

    print("#" * 60)
    print("# 曲线特征工程 + 随机森林（保留原始 pitch/vel）")
    print("#" * 60)

    X, y = load_rf_data(base, genres, 120)
    print(f"\n特征维度: {X.shape[1]}")
    print(f"样本数: {X.shape[0]}")
    print(f"特征列表: {FEATURE_NAMES}")

    # 5-fold CV
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds, all_true = [], []
    for tr, te in skf.split(X_scaled, y):
        rf.fit(X_scaled[tr], y[tr])
        all_preds.extend(rf.predict(X_scaled[te]))
        all_true.extend(y[te])

    acc = accuracy_score(all_true, all_preds)
    baseline = 1 / len(genres)

    print(f"\n{'='*50}")
    print(f"准确率: {acc:.4f} ({acc*100:.1f}%)")
    print(f"随机基线: {baseline:.4f} ({baseline*100:.1f}%)")
    print(f"{'='*50}")

    # 混淆矩阵
    cm = confusion_matrix(all_true, all_preds, labels=genres)
    print(f"\n混淆矩阵 (%):")
    header = '  '.join(f'{g[:5]:>5}' for g in genres)
    print(f"        {header}")
    for i, g in enumerate(genres):
        total = cm[i].sum() or 1
        row = '  '.join(f'{cm[i,j]/total*100:>4.0f}%' for j in range(len(genres)))
        print(f" {g[:5]:>5}  {row}")

    # 对比
    print(f"\n{'='*45}")
    print(f"{'方法':25s}  精度")
    print(f"{'-'*35}")
    print(f"{'Hausdorff (max)':25s}  33.0%")
    print(f"{'MHD (mean)':25s}  35.9%")
    print(f"{'RF (曲线特征,原始值)':25s}  {acc*100:.1f}%")
    print(f"{'='*45}")

    # 特征重要性
    rf.fit(scaler.fit_transform(X), y)
    top = np.argsort(rf.feature_importances_)[::-1][:10]
    print(f"\nTop 10 特征:")
    for i, idx in enumerate(top):
        print(f"  {i+1}. {FEATURE_NAMES[idx]}: {rf.feature_importances_[idx]:.4f}")
