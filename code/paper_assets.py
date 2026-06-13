"""Generate publication-ready Chinese figures and presentation CSV files.

The numerical experiment remains in ``final_experiment.py``.  This script only
reuses its cached data and evaluation functions to produce the final paper
assets with a consistent visual language.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyBboxPatch
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_experiment import (  # noqa: E402
    GENRES,
    distance_statistics,
    evaluate_random_forest,
    evaluate_weighted_mhd,
    relative_curve,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final"
CACHE = RESULTS / "cache"
RAW_TABLES = RESULTS / "tables"
PAPER_TABLES = RESULTS / "tables_paper"
PAPER_FIGURES = RESULTS / "figures_paper"
REPORT_FIGURES = ROOT / "report_clk" / "figures"

NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#159D9B"
GOLD = "#D9A441"
CORAL = "#D96C5F"
SLATE = "#5D6D7E"
LIGHT = "#F3F6FA"
GRID = "#D9E1EA"
TEXT = "#243447"

GENRE_CN = {
    "Classical": "古典",
    "Jazz": "爵士",
    "Rock": "摇滚",
    "Blues": "布鲁斯",
    "Electronic": "电子",
}
GENRE_COLORS = {
    "Classical": "#315C99",
    "Jazz": "#D99A3D",
    "Rock": "#C85A54",
    "Blues": "#4C9A92",
    "Electronic": "#755B9D",
}
METHOD_CN = {
    "HD local min-max": "原始 HD（逐曲归一化）",
    "HD relative TP": "HD（相对音高）",
    "HD relative TPV": "HD（三维固定权重）",
    "Q95-HD relative TPV": "Q95-HD（三维固定权重）",
    "MHD relative TPV": "MHD（三维固定权重）",
    "DTW relative pitch": "DTW（相对音高）",
    "Weighted MHD (nested)": "加权 MHD（嵌套选择）",
    "RF descriptors": "随机森林（统计描述符）",
}
FEATURE_CN = {
    "duration_beats": "曲长（拍）",
    "onset_count": "起音数量",
    "onset_density": "起音密度",
    "pitch_mean": "平均音高",
    "pitch_std": "音高标准差",
    "pitch_range": "音域",
    "pitch_q10": "音高 10% 分位",
    "pitch_q90": "音高 90% 分位",
    "interval_abs_mean": "平均绝对音程",
    "interval_abs_std": "绝对音程标准差",
    "interval_abs_max": "最大绝对音程",
    "step_fraction": "级进比例",
    "leap_fraction": "跳进比例",
    "repeat_fraction": "同音反复比例",
    "ascending_fraction": "上行比例",
    "descending_fraction": "下行比例",
    "direction_change": "旋律方向变化率",
    "velocity_mean": "平均力度",
    "velocity_std": "力度标准差",
    "velocity_range": "力度范围",
    "ioi_mean": "平均起音间隔",
    "ioi_std": "起音间隔标准差",
    "ioi_cv": "起音间隔变异系数",
    "ioi_q25": "起音间隔 25% 分位",
    "ioi_q75": "起音间隔 75% 分位",
    "offbeat_fraction": "弱拍起音比例",
    "pitch_class_entropy": "音级熵",
    "pitch_class_peak": "主音级集中度",
    "pc_0": "C 音级占比",
    "pc_1": "C♯/D♭ 音级占比",
    "pc_2": "D 音级占比",
    "pc_3": "D♯/E♭ 音级占比",
    "pc_4": "E 音级占比",
    "pc_5": "F 音级占比",
    "pc_6": "F♯/G♭ 音级占比",
    "pc_7": "G 音级占比",
    "pc_8": "G♯/A♭ 音级占比",
    "pc_9": "A 音级占比",
    "pc_10": "A♯/B♭ 音级占比",
    "pc_11": "B 音级占比",
}


def configure_style() -> None:
    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#AAB5C0",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIGURES / filename
    fig.savefig(target, dpi=300, pad_inches=0.08)
    shutil.copy2(target, REPORT_FIGURES / filename)
    plt.close(fig)


def save_csv(filename: str, rows: list[dict]) -> None:
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (PAPER_TABLES / filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_assets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    dataset_file = CACHE / "dataset_g80_p96_s2026.npz"
    distance_file = CACHE / "distances_g80_p96_s2026.npz"
    sensitivity_file = CACHE / "sensitivity_mhd_g80_s2026.npz"
    dataset_npz = np.load(dataset_file, allow_pickle=False)
    distance_npz = np.load(distance_file, allow_pickle=False)
    sensitivity_npz = np.load(sensitivity_file, allow_pickle=False)
    return (
        {key: dataset_npz[key] for key in dataset_npz.files},
        {key: distance_npz[key] for key in distance_npz.files},
        {key: sensitivity_npz[key] for key in sensitivity_npz.files},
    )


def export_tables(
    dataset: dict[str, np.ndarray],
    stats: dict,
    genre_matrix: np.ndarray,
) -> None:
    summary = pd.read_csv(RAW_TABLES / "model_summary.csv")
    model_rows = []
    for _, row in summary.iterrows():
        model_rows.append(
            {
                "方法": METHOD_CN[row["method"]],
                "总体准确率": f"{100 * row['accuracy']:.2f}%",
                "五折均值": f"{100 * row['fold_mean']:.2f}%",
                "折间标准差": f"{100 * row['fold_std']:.2f}%",
                "平衡准确率": f"{100 * row['balanced_accuracy']:.2f}%",
                "Macro-F1": f"{row['macro_f1']:.4f}",
                "各折K值": "" if pd.isna(row["k_selected"]) else str(row["k_selected"]),
                "各折音量权重": ""
                if pd.isna(row.get("weight_selected"))
                else str(row["weight_selected"]),
            }
        )
    save_csv("模型综合结果.csv", model_rows)

    folds = pd.read_csv(RAW_TABLES / "fold_results.csv")
    fold_rows = []
    for _, row in folds.iterrows():
        fold_rows.append(
            {
                "方法": METHOD_CN[row["method"]],
                "外层折": int(row["fold"]),
                "K值": "" if pd.isna(row["k"]) else int(row["k"]),
                "测试样本数": int(row["n_test"]),
                "准确率": f"{100 * row['accuracy']:.2f}%",
                "平衡准确率": f"{100 * row['balanced_accuracy']:.2f}%",
                "Macro-F1": f"{row['macro_f1']:.4f}",
                "音量权重": ""
                if pd.isna(row.get("velocity_weight"))
                else f"{row['velocity_weight']:.2f}",
            }
        )
    save_csv("外层五折结果.csv", fold_rows)

    matrix_rows = []
    for i, genre in enumerate(GENRES):
        row = {"真实曲风": GENRE_CN[genre]}
        row.update(
            {
                GENRE_CN[column_genre]: f"{genre_matrix[i, j]:.4f}"
                for j, column_genre in enumerate(GENRES)
            }
        )
        matrix_rows.append(row)
    save_csv("曲风平均距离矩阵.csv", matrix_rows)

    sensitivity = pd.read_csv(RAW_TABLES / "sensitivity.csv")
    sensitivity_rows = []
    parameter_cn = {"resample_points": "重采样点数", "velocity_weight": "音量权重"}
    for _, row in sensitivity.iterrows():
        sensitivity_rows.append(
            {
                "参数": parameter_cn[row["parameter"]],
                "取值": f"{row['value']:g}",
                "五折准确率均值": f"{100 * row['fold_mean']:.2f}%",
                "折间标准差": f"{100 * row['fold_std']:.2f}%",
            }
        )
    save_csv("敏感性分析.csv", sensitivity_rows)

    save_csv(
        "统计检验与核心结论.csv",
        [
            {
                "同曲风MHD均值": f"{stats['within_mean']:.4f}",
                "异曲风MHD均值": f"{stats['between_mean']:.4f}",
                "均值差": f"{stats['mean_gap']:.4f}",
                "配对AUC": f"{stats['pair_auc']:.4f}",
                "置换检验p值": f"{stats['permutation_p']:.4f}",
                "结论": "差异显著，但效应量有限",
            }
        ],
    )

    labels = dataset["labels"]
    sample_rows = []
    for genre in GENRES:
        mask = labels == genre
        sample_rows.append(
            {
                "曲风": GENRE_CN[genre],
                "英文标签": genre,
                "样本数": int(mask.sum()),
                "样本占比": f"{100 * mask.mean():.1f}%",
            }
        )
    sample_rows.append(
        {
            "曲风": "合计",
            "英文标签": "Total",
            "样本数": int(len(labels)),
            "样本占比": "100.0%",
        }
    )
    save_csv("样本构成.csv", sample_rows)

    robustness = pd.read_csv(RAW_TABLES / "synthetic_robustness.csv")
    robustness_rows = [
        {
            "单点扰动幅度": f"{row['outlier_amplitude']:.1f}",
            "max-HD": f"{row['hd']:.4f}",
            "Q95-HD": f"{row['q95']:.4f}",
            "MHD": f"{row['mhd']:.4f}",
        }
        for _, row in robustness.iterrows()
    ]
    save_csv("离群点稳健性.csv", robustness_rows)


def plot_example_curves(dataset: dict[str, np.ndarray]) -> None:
    curves = dataset["curves"]
    labels = dataset["labels"]
    fig = plt.figure(figsize=(12.4, 7.1))
    for index, genre in enumerate(GENRES, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        curve = relative_curve(curves[np.flatnonzero(labels == genre)[0]], 0.25)
        color = GENRE_COLORS[genre]
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, lw=2.0)
        ax.scatter(
            curve[::8, 0],
            curve[::8, 1],
            curve[::8, 2],
            s=14,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            depthshade=False,
        )
        ax.set_title(f"{GENRE_CN[genre]}（{genre}）", pad=5)
        ax.set_xlabel("时间相位", labelpad=4)
        ax.set_ylabel("相对音高", labelpad=4)
        ax.grid(True, alpha=0.25)
        ax.view_init(elev=22, azim=-57)

    note = fig.add_subplot(2, 3, 6)
    note.axis("off")
    note.add_patch(
        FancyBboxPatch(
            (0.07, 0.17),
            0.86,
            0.66,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            transform=note.transAxes,
            facecolor=LIGHT,
            edgecolor="#CBD6E2",
        )
    )
    note.text(0.14, 0.68, "统一三维表示", fontsize=14, weight="bold", color=NAVY)
    note.text(
        0.14,
        0.56,
        "横轴：归一化时间相位 $t$",
        fontsize=11,
        transform=note.transAxes,
    )
    note.text(
        0.14,
        0.44,
        "纵轴：移调不变的相对音高",
        fontsize=11,
        transform=note.transAxes,
    )
    note.text(
        0.14,
        0.32,
        "竖轴：中心化后的加权力度",
        fontsize=11,
        transform=note.transAxes,
    )
    note.text(
        0.14,
        0.21,
        "每条曲线均匀重采样为 96 点",
        fontsize=10,
        color=SLATE,
        transform=note.transAxes,
    )
    fig.suptitle("五类曲风的三维旋律线示例", fontsize=17, weight="bold", color=NAVY)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.90, wspace=0.15, hspace=0.18)
    save_figure(fig, "example_curves_cn.png")


def plot_synthetic() -> None:
    data = pd.read_csv(RAW_TABLES / "synthetic_robustness.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    series = [
        ("hd", "max-HD", CORAL, "o"),
        ("q95", "Q95-HD", GOLD, "s"),
        ("mhd", "MHD", TEAL, "^"),
    ]
    for key, label, color, marker in series:
        ax.plot(
            data["outlier_amplitude"],
            data[key],
            label=label,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=2.2,
        )
    ax.fill_between(
        data["outlier_amplitude"],
        data["mhd"],
        color=TEAL,
        alpha=0.08,
    )
    ax.annotate(
        "最大值型距离被单个异常点迅速主导",
        xy=(1.55, data.loc[data["outlier_amplitude"] == 1.6, "hd"].iloc[0]),
        xytext=(0.72, 1.35),
        arrowprops={"arrowstyle": "->", "color": CORAL, "lw": 1.2},
        color=CORAL,
        fontsize=10,
    )
    ax.set_title("单点离群扰动下 Hausdorff 距离族的稳健性")
    ax.set_xlabel("单点扰动幅度")
    ax.set_ylabel("与原曲线的距离")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean_axis(ax)
    save_figure(fig, "synthetic_robustness_cn.png")


def plot_distance_distribution(stats: dict) -> None:
    within = stats["within"]
    between = stats["between"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [1.65, 1]})
    bins = np.linspace(min(within.min(), between.min()), np.quantile(between, 0.995), 55)
    axes[0].hist(
        within,
        bins=bins,
        density=True,
        color=TEAL,
        alpha=0.58,
        label=f"同曲风（均值 {stats['within_mean']:.3f}）",
    )
    axes[0].hist(
        between,
        bins=bins,
        density=True,
        color=GOLD,
        alpha=0.50,
        label=f"异曲风（均值 {stats['between_mean']:.3f}）",
    )
    axes[0].axvline(stats["within_mean"], color=TEAL, lw=1.8)
    axes[0].axvline(stats["between_mean"], color=GOLD, lw=1.8)
    axes[0].set_title("距离分布：异类整体右移但重叠明显")
    axes[0].set_xlabel("Modified Hausdorff Distance")
    axes[0].set_ylabel("概率密度")
    axes[0].legend(frameon=False)
    clean_axis(axes[0])

    parts = axes[1].violinplot(
        [within, between],
        positions=[1, 2],
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.75,
    )
    for body, color in zip(parts["bodies"], [TEAL, GOLD]):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.72)
    parts["cmedians"].set_color(NAVY)
    parts["cmedians"].set_linewidth(2)
    axes[1].boxplot(
        [within, between],
        positions=[1, 2],
        widths=0.22,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": NAVY},
        medianprops={"color": NAVY, "linewidth": 1.5},
        whiskerprops={"color": NAVY},
        capprops={"color": NAVY},
    )
    axes[1].set_xticks([1, 2], ["同曲风", "异曲风"])
    axes[1].set_ylabel("MHD")
    axes[1].set_title(f"置换检验 $p={stats['permutation_p']:.3f}$\n配对 AUC={stats['pair_auc']:.3f}")
    clean_axis(axes[1])
    fig.suptitle("同曲风与异曲风旋律的 MHD 差异", fontsize=16, weight="bold", color=NAVY)
    fig.tight_layout()
    save_figure(fig, "distance_distribution_cn.png")


def plot_heatmap(genre_matrix: np.ndarray) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "paper_teal", ["#F4F7FA", "#BFD9D7", "#52A5A2", "#17365D"]
    )
    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    image = ax.imshow(genre_matrix, cmap=cmap, vmin=0.26, vmax=0.40)
    labels = [GENRE_CN[genre] for genre in GENRES]
    ax.set_xticks(range(len(GENRES)), labels)
    ax.set_yticks(range(len(GENRES)), labels)
    ax.set_xlabel("曲风 B")
    ax.set_ylabel("曲风 A")
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            color = "white" if genre_matrix[i, j] > 0.35 else TEXT
            weight = "bold" if i == j else "normal"
            ax.text(
                j,
                i,
                f"{genre_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                weight=weight,
            )
    for index in range(len(GENRES)):
        ax.add_patch(
            plt.Rectangle(
                (index - 0.49, index - 0.49),
                0.98,
                0.98,
                fill=False,
                edgecolor=GOLD,
                linewidth=2,
            )
        )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.04)
    colorbar.set_label("平均 MHD（越小越相似）")
    ax.set_title("五类曲风的平均 MHD 矩阵\n金色方框表示类内距离")
    fig.tight_layout()
    save_figure(fig, "genre_distance_heatmap_cn.png")


def add_covariance_ellipse(ax, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * 1.55 * np.sqrt(np.maximum(values, 0))
    ellipse = Ellipse(
        points.mean(axis=0),
        width,
        height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.10,
        linewidth=1.4,
    )
    ax.add_patch(ellipse)


def plot_mds(matrix: np.ndarray, labels: np.ndarray) -> None:
    embedding = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=2,
        max_iter=300,
    ).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for genre in GENRES:
        mask = labels == genre
        points = embedding[mask]
        add_covariance_ellipse(ax, points, GENRE_COLORS[genre])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=25,
            alpha=0.72,
            color=GENRE_COLORS[genre],
            edgecolor="white",
            linewidth=0.35,
            label=GENRE_CN[genre],
        )
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("MDS 维度 1")
    ax.set_ylabel("MDS 维度 2")
    ax.set_title("MHD 距离矩阵的二维投影\n半透明椭圆表示各曲风的主要分布范围")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    clean_axis(ax, grid_axis="both")
    save_figure(fig, "mds_mhd_cn.png")


def plot_model_comparison() -> None:
    data = pd.read_csv(RAW_TABLES / "model_summary.csv")
    data["方法中文"] = data["method"].map(METHOD_CN)
    order = [
        "HD local min-max",
        "HD relative TP",
        "HD relative TPV",
        "Q95-HD relative TPV",
        "MHD relative TPV",
        "DTW relative pitch",
        "Weighted MHD (nested)",
        "RF descriptors",
    ]
    data = data.set_index("method").loc[order].reset_index()
    colors = [
        "#98A6B5",
        "#7EA6CE",
        "#5E91C5",
        "#49A6A2",
        TEAL,
        "#9B83B5",
        GOLD,
        NAVY,
    ]
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    y = np.arange(len(data))
    bars = ax.barh(
        y,
        100 * data["fold_mean"],
        xerr=100 * data["fold_std"],
        color=colors,
        height=0.66,
        capsize=3,
        error_kw={"ecolor": SLATE, "lw": 1.1},
    )
    ax.axvline(20, color=CORAL, linestyle="--", linewidth=1.4)
    ax.text(
        20.6,
        0.02,
        "随机基线 20%",
        transform=ax.get_xaxis_transform(),
        color=CORAL,
        fontsize=9,
        va="bottom",
    )
    ax.set_yticks(y, data["方法中文"])
    ax.invert_yaxis()
    ax.set_xlabel("分组五折准确率（%）")
    ax.set_xlim(0, 66)
    ax.set_title("不同距离模型与能力对照的分类表现")
    for bar, mean, std in zip(bars, data["fold_mean"], data["fold_std"]):
        ax.text(
            100 * mean + 100 * std + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{100 * mean:.2f}%",
            va="center",
            fontsize=9.5,
            weight="bold" if mean >= 0.40 else "normal",
            color=NAVY if mean >= 0.40 else TEXT,
        )
    ax.text(
        0.99,
        0.02,
        "误差线：五个外层测试折的样本标准差",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=SLATE,
    )
    clean_axis(ax, grid_axis="x")
    fig.tight_layout()
    save_figure(fig, "model_comparison_cn.png")


def draw_confusion(ax, labels: np.ndarray, predictions: np.ndarray, title: str, accuracy: float) -> None:
    cm = confusion_matrix(labels, predictions, labels=GENRES, normalize="true")
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue", ["#F5F8FC", "#AFC9E2", "#2F75B5", "#17365D"]
    )
    image = ax.imshow(cm, cmap=cmap, vmin=0, vmax=0.75)
    names = [GENRE_CN[genre] for genre in GENRES]
    ax.set_xticks(range(len(GENRES)), names, rotation=28, ha="right")
    ax.set_yticks(range(len(GENRES)), names)
    ax.set_xlabel("预测曲风")
    ax.set_ylabel("真实曲风")
    ax.set_title(f"{title}\n总体准确率 {100 * accuracy:.1f}%")
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            value = cm[i, j]
            ax.text(
                j,
                i,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                color="white" if value > 0.42 else TEXT,
                weight="bold" if i == j else "normal",
                fontsize=9.5,
            )
    return image


def plot_confusions(
    labels: np.ndarray,
    weighted_predictions: np.ndarray,
    rf_predictions: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2))
    draw_confusion(axes[0], labels, weighted_predictions, "加权 MHD（嵌套）", 0.40)
    draw_confusion(axes[1], labels, rf_predictions, "随机森林（统计描述符）", 0.57)
    fig.suptitle("两类模型的折外混淆结构", fontsize=16, weight="bold", color=NAVY)
    fig.text(
        0.5,
        0.025,
        "矩阵按真实类别逐行归一化，单元格为折外预测比例",
        ha="center",
        fontsize=9,
        color=SLATE,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.84, wspace=0.28)
    save_figure(fig, "confusion_models_cn.png")


def plot_sensitivity() -> None:
    data = pd.read_csv(RAW_TABLES / "sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    settings = [
        ("resample_points", "重采样点数", "曲线离散密度"),
        ("velocity_weight", "音量权重 $w_v$", "力度维权重"),
    ]
    for ax, (parameter, xlabel, title) in zip(axes, settings):
        selected = data[data["parameter"] == parameter]
        x = selected["value"].to_numpy(dtype=float)
        y = 100 * selected["fold_mean"].to_numpy(dtype=float)
        error = 100 * selected["fold_std"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            markersize=7,
            capsize=4,
            color=TEAL,
            ecolor="#7F98AA",
            linewidth=2,
        )
        ax.axhline(20, color=CORAL, linestyle="--", linewidth=1.2)
        ax.fill_between(x, 20, y, color=TEAL, alpha=0.06)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("分组五折准确率（%）")
        ax.set_ylim(15, 49)
        for x_value, y_value in zip(x, y):
            ax.annotate(
                f"{y_value:.1f}%",
                (x_value, y_value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        clean_axis(ax)
    fig.suptitle("MHD 模型的参数敏感性", fontsize=16, weight="bold", color=NAVY)
    fig.tight_layout()
    save_figure(fig, "sensitivity_cn.png")


def plot_feature_importance(importances: np.ndarray, names: np.ndarray) -> None:
    top = np.argsort(importances)[-12:]
    values = importances[top]
    labels = [FEATURE_CN.get(str(name), str(name)) for name in names[top]]
    colors = [TEAL] * len(top)
    colors[-3:] = [GOLD, GOLD, NAVY]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(np.arange(len(top)), values, color=colors, height=0.68)
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel("五折平均特征重要性")
    ax.set_title("随机森林最重要的 12 个音乐描述符")
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.8,
        )
    ax.set_xlim(0, values.max() * 1.22)
    clean_axis(ax, grid_axis="x")
    fig.tight_layout()
    save_figure(fig, "feature_importance_cn.png")


def plot_dashboard(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 3.25))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.25)
    ax.axis("off")
    cards = [
        ("400", "五类平衡样本", NAVY),
        ("40.05%", "最佳几何模型准确率", TEAL),
        ("56.99%", "多特征能力对照", GOLD),
        ("0.002", "标签置换检验 $p$ 值", CORAL),
    ]
    for index, (value, label, color) in enumerate(cards):
        x = 0.25 + index * 2.95
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.42),
                2.55,
                1.95,
                boxstyle="round,pad=0.025,rounding_size=0.08",
                facecolor="white",
                edgecolor="#D4DEE8",
                linewidth=1.1,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.42),
                0.12,
                1.95,
                boxstyle="round,pad=0.0,rounding_size=0.05",
                facecolor=color,
                edgecolor=color,
            )
        )
        ax.text(x + 0.28, 1.62, value, fontsize=24, weight="bold", color=color)
        ax.text(x + 0.28, 1.10, label, fontsize=10.5, color=TEXT)
    ax.text(
        0.28,
        2.86,
        "核心结果一览",
        fontsize=17,
        weight="bold",
        color=NAVY,
    )
    ax.text(
        11.72,
        2.87,
        f"同类/异类均值差 {stats['mean_gap']:.4f} · 配对 AUC {stats['pair_auc']:.3f}",
        fontsize=9.5,
        color=SLATE,
        ha="right",
    )
    save_figure(fig, "results_dashboard_cn.png")


def main() -> None:
    configure_style()
    dataset, matrices, sensitivity = load_assets()
    labels = dataset["labels"]
    groups = dataset["groups"]

    stats = distance_statistics(matrices["mhd_tpv"], labels, seed=2026)
    genre_matrix = np.loadtxt(
        RAW_TABLES / "genre_mean_distance.csv", delimiter=",", skiprows=1
    )
    velocity_matrices = {
        weight: sensitivity[f"velocity_{weight:.2f}"]
        for weight in (0.0, 0.10, 0.25, 0.50)
    }
    weighted_summary, weighted_predictions, _ = evaluate_weighted_mhd(
        velocity_matrices, labels, groups
    )
    rf_summary, rf_predictions, _, importances = evaluate_random_forest(
        dataset["features"], labels, groups
    )

    export_tables(dataset, stats, genre_matrix)
    plot_dashboard(stats)
    plot_example_curves(dataset)
    plot_synthetic()
    plot_distance_distribution(stats)
    plot_heatmap(genre_matrix)
    plot_model_comparison()
    plot_confusions(labels, weighted_predictions, rf_predictions)
    plot_mds(matrices["mhd_tpv"], labels)
    plot_feature_importance(importances, dataset["feature_names"])
    plot_sensitivity()

    print(
        "[paper-assets] weighted MHD accuracy="
        f"{weighted_summary['accuracy']:.3f}, RF accuracy={rf_summary['accuracy']:.3f}"
    )
    print(f"[paper-assets] figures: {PAPER_FIGURES}")
    print(f"[paper-assets] tables: {PAPER_TABLES}")


if __name__ == "__main__":
    main()
