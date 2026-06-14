"""Generate synchronized Chinese tables and publication-ready figures."""

from __future__ import annotations

import csv
import hashlib
import json
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
from data_pipeline import GENRES, RESULTS_DIR, ROOT, dataset_cache_path  # noqa: E402
from distance_models import distance_cache_path  # noqa: E402
from midi_geometry import relative_curve  # noqa: E402


RAW_TABLES = RESULTS_DIR / "tables"
PAPER_TABLES = RESULTS_DIR / "tables_paper"
PAPER_FIGURES = RESULTS_DIR / "figures_paper"
REPORT_FIGURES = ROOT / "report_clk" / "figures"
SUMMARY_FILE = RESULTS_DIR / "summary.json"
PREDICTION_FILE = RESULTS_DIR / "predictions_primary.npz"
METHOD_FIGURE = REPORT_FIGURES / "method_taxonomy_cn.png"
METHOD_PROMPT = REPORT_FIGURES / "method_taxonomy_cn.prompt.txt"

NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#159D9B"
GOLD = "#D9A441"
CORAL = "#D96C5F"
PURPLE = "#755B9D"
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
    "Phase-aligned trajectory RMSE": "相位对齐轨迹 RMSE",
    "Tuned MHD (nested)": "多参数 MHD（嵌套）",
    "Multivariate DTW (nested)": "多变量 DTW（嵌套）",
    "Multinomial logistic descriptors": "多项逻辑回归（描述符）",
    "RF descriptors": "随机森林（描述符）",
    "MHD-RF probability fusion": "MHD-RF 概率融合",
}
FEATURE_CN = {
    "duration_beats": "曲长",
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
    "direction_change": "方向变化率",
    "contour_slope": "旋律轮廓斜率",
    "pitch_autocorrelation": "音高一阶相关",
    "interval_entropy": "音程熵",
    "velocity_mean": "平均力度",
    "velocity_std": "力度标准差",
    "velocity_range": "力度范围",
    "velocity_change_abs_mean": "平均力度变化",
    "velocity_change_std": "力度变化标准差",
    "pitch_velocity_correlation": "音高-力度相关",
    "ioi_mean": "平均起音间隔",
    "ioi_std": "起音间隔标准差",
    "ioi_cv": "起音间隔变异系数",
    "ioi_q25": "起音间隔 25% 分位",
    "ioi_q75": "起音间隔 75% 分位",
    "offbeat_fraction": "弱拍起音比例",
    "rhythm_regularity": "节奏规则度",
    "ioi_entropy": "起音间隔熵",
    "pitch_class_entropy": "音级熵",
    "pitch_class_peak": "主音级集中度",
    "pitch_class_top3": "前三音级集中度",
}


def configure_style() -> None:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_method_taxonomy(
    source: Path = METHOD_FIGURE,
    prompt_file: Path = METHOD_PROMPT,
    output_dir: Path = PAPER_FIGURES,
) -> dict:
    """Record and copy the curated method graphic without modifying its source."""
    if not source.exists():
        raise FileNotFoundError(f"curated method figure is missing: {source}")
    if not prompt_file.exists():
        raise FileNotFoundError(f"method figure prompt is missing: {prompt_file}")
    image = plt.imread(source)
    height, width = image.shape[:2]
    try:
        source_label = str(source.relative_to(ROOT))
    except ValueError:
        source_label = str(source)
    try:
        prompt_label = str(prompt_file.relative_to(ROOT))
    except ValueError:
        prompt_label = str(prompt_file)
    metadata = {
        "source": source_label,
        "prompt_file": prompt_label,
        "width_px": int(width),
        "height_px": int(height),
        "sha256": sha256_file(source),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_dir / source.name)
    shutil.copy2(prompt_file, output_dir / prompt_file.name)
    (output_dir / "method_taxonomy_cn.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def save_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (PAPER_TABLES / filename).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_key(method: str) -> str:
    return (
        method.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_assets() -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    config = summary["config"]
    dataset_file = dataset_cache_path(
        int(config["requested_per_genre"]),
        int(config["base_points"]),
        int(config["seed"]),
    )
    distance_file = distance_cache_path(
        int(summary["per_genre_actual"]),
        int(config["base_points"]),
        int(config["seed"]),
    )
    dataset_npz = np.load(dataset_file, allow_pickle=False)
    distance_npz = np.load(distance_file, allow_pickle=False)
    prediction_npz = np.load(PREDICTION_FILE, allow_pickle=False)
    return (
        summary,
        {key: dataset_npz[key] for key in dataset_npz.files},
        {key: distance_npz[key] for key in distance_npz.files},
        {key: prediction_npz[key] for key in prediction_npz.files},
    )


def distance_arrays(summary: dict, matrix: np.ndarray, labels: np.ndarray) -> dict:
    upper = np.triu_indices(len(labels), k=1)
    same = labels[upper[0]] == labels[upper[1]]
    return {
        **summary["distance_statistics_mhd"],
        "within": matrix[upper][same],
        "between": matrix[upper][~same],
    }


def export_tables(summary: dict, dataset: dict, predictions: dict) -> None:
    model_rows = []
    for row in summary["models"]:
        model_rows.append(
            {
                "方法": METHOD_CN.get(row["method"], row["method"]),
                "总体准确率": f"{100 * row['accuracy']:.2f}%",
                "准确率95%CI": (
                    f"[{100 * row['accuracy_ci_low']:.2f}%, "
                    f"{100 * row['accuracy_ci_high']:.2f}%]"
                ),
                "平衡准确率": f"{100 * row['balanced_accuracy']:.2f}%",
                "Macro-F1": f"{row['macro_f1']:.4f}",
                "五折标准差": f"{100 * row['fold_std']:.2f}%",
                "选择参数": "; ".join(
                    f"{key}={row[key]}"
                    for key in (
                        "k_selected",
                        "resample_points_selected",
                        "velocity_weight_selected",
                        "window_fraction_selected",
                        "c_selected",
                        "rf_weight_selected",
                    )
                    if row.get(key, "") not in ("", None)
                ),
            }
        )
    save_csv("模型综合结果.csv", model_rows)

    for source, target in (
        ("data_audit_summary.csv", "数据筛选审计.csv"),
        ("cross_genre_conflict_summary.csv", "跨曲风冲突组摘要.csv"),
        ("feature_ablation_repeated.csv", "重复特征组消融.csv"),
        ("feature_group_importance.csv", "组级置换重要性.csv"),
        ("mcnemar_holm.csv", "模型配对检验.csv"),
        ("class_recall.csv", "分曲风召回率.csv"),
        ("sample_split_validation_summary.csv", "样本与分组重复验证.csv"),
        ("paired_model_differences.csv", "完整运行配对差异.csv"),
    ):
        rows = pd.read_csv(RAW_TABLES / source).to_dict("records")
        save_csv(target, rows)

    diagnostics = pd.read_csv(RAW_TABLES / "model_disagreements.csv")
    save_csv("典型误分类与模型分歧.csv", diagnostics.head(30).to_dict("records"))

    importances = predictions["rf_importances"]
    names = predictions["feature_names"]
    order = np.argsort(importances)[::-1]
    save_csv(
        "随机森林折外置换重要性.csv",
        [
            {
                "排名": rank,
                "特征": FEATURE_CN.get(str(names[index]), str(names[index])),
                "英文特征名": str(names[index]),
                "重要性": f"{importances[index]:.6f}",
            }
            for rank, index in enumerate(order, start=1)
        ],
    )

    labels = dataset["labels"]
    save_csv(
        "样本构成.csv",
        [
            {
                "曲风": GENRE_CN[genre],
                "英文标签": genre,
                "样本数": int(np.sum(labels == genre)),
                "比例": f"{100 * np.mean(labels == genre):.1f}%",
            }
            for genre in GENRES
        ],
    )


def plot_data_funnel(summary: dict) -> None:
    data = pd.read_csv(RAW_TABLES / "data_audit_summary.csv")
    data = data[data["genre"].isin(GENRES)].copy()
    labels = [GENRE_CN[value] for value in data["genre"]]
    raw = data["raw_files"].to_numpy(float)
    candidates = data["candidate_groups_after_dedup"].to_numpy(float)
    selected = data["selected_samples"].to_numpy(float)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.3, 5.1))
    ax.barh(y, raw, color="#D8E0E8", height=0.72, label="原始 MIDI 文件")
    ax.barh(y, candidates, color=TEAL, height=0.50, label="去重且排除跨曲风冲突后的组")
    ax.barh(y, selected, color=GOLD, height=0.28, label="最终平衡样本")
    for index, value in enumerate(selected):
        ax.text(
            max(1.2, value * 0.72),
            index,
            f"{int(value)}",
            va="center",
            ha="center",
            weight="bold",
            color="white",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    genre_font_path = Path("C:/Windows/Fonts/simhei.ttf")
    genre_font = (
        font_manager.FontProperties(fname=str(genre_font_path), size=11)
        if genre_font_path.exists()
        else font_manager.FontProperties(family="sans-serif", size=11)
    )
    for index, label in enumerate(labels):
        ax.annotate(
            f"{label}\u3000",
            xy=(0, index),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=TEXT,
            fontproperties=genre_font,
            annotation_clip=False,
        )
    ax.invert_yaxis()
    ax.set_xlabel("文件或独立作品组数量（对数坐标）")
    ax.set_xscale("log")
    ax.set_title(
        f"数据筛选漏斗：三重指纹去重后形成 {summary['n_samples']} 首平衡样本"
    )
    ax.legend(frameon=False, ncol=3, loc="lower right")
    clean_axis(ax, "x")
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.16, top=0.88)
    save_figure(fig, "data_funnel_cn.png")


def plot_example_curves(dataset: dict) -> None:
    curves = dataset["curves"]
    labels = dataset["labels"]
    examples = [
        relative_curve(curves[np.flatnonzero(labels == genre)[0]], 0.50)
        for genre in GENRES
    ]
    velocity_limit = max(float(np.max(np.abs(curve[:, 2]))) for curve in examples)
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.5), sharex=True, sharey=True)
    scatter = None
    for ax, genre, curve in zip(axes.flat, GENRES, examples):
        color = GENRE_COLORS[genre]
        ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.7, alpha=0.88)
        scatter = ax.scatter(
            curve[:, 0],
            curve[:, 1],
            c=curve[:, 2],
            cmap="coolwarm",
            vmin=-velocity_limit,
            vmax=velocity_limit,
            s=22,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.set_title(f"{GENRE_CN[genre]}（{genre}）")
        ax.set_xlabel("归一化时间相位")
        ax.set_ylabel("相对中位音高（八度）")
        clean_axis(ax, "both")
    note_slot = axes.flat[-1].get_position()
    axes.flat[-1].remove()
    note = fig.add_axes(note_slot)
    note.set_xlim(0, 1)
    note.set_ylim(0, 1)
    note.axis("off")
    note.add_patch(
        FancyBboxPatch(
            (0.08, 0.16),
            0.84,
            0.68,
            boxstyle="round,pad=0.03",
            facecolor=LIGHT,
            edgecolor="#CBD6E2",
        )
    )
    note.text(0.15, 0.69, "二维投影与力度编码", fontsize=14, weight="bold", color=NAVY)
    note.text(0.15, 0.54, "横轴：统一的时间相位", fontsize=10.5)
    note.text(0.15, 0.42, "纵轴：移调不变的相对音高", fontsize=10.5)
    note.text(0.15, 0.30, "颜色：中心化后的力度动态", fontsize=10.5)
    if scatter is not None:
        color_axis = note.inset_axes([0.15, 0.21, 0.70, 0.055])
        colorbar = fig.colorbar(
            scatter,
            cax=color_axis,
            orientation="horizontal",
        )
        colorbar.set_label("低力度 ← 加权力度 → 高力度", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "五类曲风的标准化旋律线示例",
        fontsize=16,
        weight="bold",
        color=NAVY,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.96,
        bottom=0.09,
        top=0.89,
        wspace=0.24,
        hspace=0.34,
    )
    save_figure(fig, "example_curves_cn.png")


def plot_synthetic() -> None:
    data = pd.read_csv(RAW_TABLES / "synthetic_robustness.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for key, label, color, marker in (
        ("hd", "max-HD", CORAL, "o"),
        ("q95", "Q95-HD", GOLD, "s"),
        ("mhd", "MHD", TEAL, "^"),
    ):
        ax.plot(
            data["outlier_amplitude"],
            data[key],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
        )
    ax.set_title("单点离群扰动下 Hausdorff 距离族的稳健性")
    ax.set_xlabel("单点扰动幅度")
    ax.set_ylabel("与原曲线的距离")
    ax.legend(frameon=False, ncol=3)
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, "synthetic_robustness_cn.png")


def plot_distance_distribution(stats: dict) -> None:
    within, between = stats["within"], stats["between"]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [1.75, 1]}
    )
    bins = np.linspace(
        min(within.min(), between.min()), np.quantile(between, 0.995), 55
    )
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
    axes[0].set_title("距离分布：异类整体右移，但重叠仍然明显")
    axes[0].set_xlabel("Modified Hausdorff Distance")
    axes[0].set_ylabel("概率密度")
    axes[0].legend(frameon=False)
    clean_axis(axes[0])

    axes[1].axis("off")
    axes[1].add_patch(
        FancyBboxPatch(
            (0.04, 0.08),
            0.92,
            0.84,
            transform=axes[1].transAxes,
            boxstyle="round,pad=0.035,rounding_size=0.04",
            facecolor=LIGHT,
            edgecolor="#CBD6E2",
            linewidth=1.2,
        )
    )
    axes[1].text(
        0.50,
        0.74,
        f"AUC = {stats['pair_auc']:.4f}",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color=NAVY,
    )
    axes[1].text(
        0.50,
        0.52,
        f"异类－同类均值差 = {stats['mean_gap']:.4f}",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=12,
        weight="bold",
        color=TEAL,
    )
    axes[1].text(
        0.50,
        0.34,
        "效应方向稳定，但判别能力较弱",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=11,
        color=TEXT,
    )
    axes[1].text(
        0.50,
        0.19,
        f"作品组置换检验：$p={stats['permutation_p']:.4f}$",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9.5,
        color=SLATE,
    )
    fig.suptitle(
        "同曲风与异曲风旋律的 MHD 差异",
        fontsize=16,
        weight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    save_figure(fig, "distance_distribution_cn.png")


def plot_heatmap(matrix: np.ndarray) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "paper_teal", ["#F4F7FA", "#BFD9D7", "#52A5A2", NAVY]
    )
    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    image = ax.imshow(matrix, cmap=cmap)
    labels = [GENRE_CN[genre] for genre in GENRES]
    ax.set_xticks(range(len(GENRES)), labels)
    ax.set_yticks(range(len(GENRES)), labels)
    for row in range(len(GENRES)):
        for column in range(len(GENRES)):
            threshold = np.quantile(matrix, 0.60)
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else TEXT,
                weight="bold" if row == column else "normal",
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
    fig.colorbar(image, ax=ax, label="平均 MHD（越小越相似）")
    ax.set_xlabel("曲风 B")
    ax.set_ylabel("曲风 A")
    ax.set_title("五类曲风的平均 MHD 矩阵")
    fig.tight_layout()
    save_figure(fig, "genre_distance_heatmap_cn.png")


def add_covariance_ellipse(ax, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 3.1 * np.sqrt(np.maximum(values, 0))
    ax.add_patch(
        Ellipse(
            points.mean(axis=0),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
        )
    )


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
        points = embedding[labels == genre]
        add_covariance_ellipse(ax, points, GENRE_COLORS[genre])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=24,
            alpha=0.70,
            color=GENRE_COLORS[genre],
            edgecolor="white",
            linewidth=0.35,
            label=GENRE_CN[genre],
        )
    ax.set_xlabel("MDS 维度 1")
    ax.set_ylabel("MDS 维度 2")
    ax.set_title("MHD 距离矩阵的二维投影")
    ax.legend(frameon=False, ncol=3)
    clean_axis(ax, "both")
    fig.tight_layout()
    save_figure(fig, "mds_mhd_cn.png")


def plot_model_comparison(summary: dict) -> None:
    preferred_order = [
        "HD local min-max",
        "Q95-HD relative TPV",
        "MHD relative TPV",
        "Phase-aligned trajectory RMSE",
        "Tuned MHD (nested)",
        "Multivariate DTW (nested)",
        "Multinomial logistic descriptors",
        "RF descriptors",
        "MHD-RF probability fusion",
    ]
    rows = {row["method"]: row for row in summary["models"]}
    data = [rows[name] for name in preferred_order]
    means = np.asarray([100 * row["accuracy"] for row in data])
    lower = means - np.asarray([100 * row["accuracy_ci_low"] for row in data])
    upper = np.asarray([100 * row["accuracy_ci_high"] for row in data]) - means
    colors = ["#AAB4BF"] * len(data)
    for index, row in enumerate(data):
        colors[index] = {
            "Tuned MHD (nested)": GOLD,
            "RF descriptors": NAVY,
            "MHD-RF probability fusion": CORAL,
        }.get(row["method"], "#AAB4BF")
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    y = np.arange(len(data))
    bars = ax.barh(
        y,
        means,
        xerr=np.vstack((lower, upper)),
        color=colors,
        height=0.66,
        capsize=3,
        error_kw={"ecolor": SLATE, "lw": 1.1},
    )
    ax.axvline(20, color=CORAL, linestyle="--", linewidth=1.3)
    ax.set_yticks(y, [METHOD_CN[row["method"]] for row in data])
    ax.invert_yaxis()
    ax.set_xlabel("主实验折外准确率（%，误差线为条件 Bootstrap 95% CI）")
    ax.set_title("距离模型、描述符模型与概率融合的统一比较")
    for bar, value in zip(bars, means):
        ax.text(
            value + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            weight="bold" if value == means.max() else "normal",
        )
    ax.set_xlim(0, max(70, float((means + upper).max() + 7)))
    clean_axis(ax, "x")
    fig.tight_layout()
    save_figure(fig, "model_comparison_cn.png")


def draw_confusion(ax, labels: np.ndarray, values: np.ndarray, title: str) -> None:
    matrix = confusion_matrix(labels, values, labels=GENRES, normalize="true")
    cmap = LinearSegmentedColormap.from_list(
        "paper_blue", ["#F5F8FC", "#AFC9E2", BLUE, NAVY]
    )
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(0.7, matrix.max()))
    names = [GENRE_CN[genre] for genre in GENRES]
    ax.set_xticks(range(len(GENRES)), names, rotation=28, ha="right")
    ax.set_yticks(range(len(GENRES)), names)
    ax.set_xlabel("预测曲风")
    ax.set_ylabel("真实曲风")
    ax.set_title(title)
    for row in range(len(GENRES)):
        for column in range(len(GENRES)):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                color="white" if value > 0.42 else TEXT,
                weight="bold" if row == column else "normal",
                fontsize=8.8,
            )


def plot_confusions(labels: np.ndarray, predictions: dict) -> None:
    mhd_method = "Tuned MHD (nested)"
    rf_method = "RF descriptors"
    fig, axes = plt.subplots(
        1, 3, figsize=(13.6, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.82]}
    )
    draw_confusion(axes[0], labels, predictions[mhd_method], METHOD_CN[mhd_method])
    draw_confusion(axes[1], labels, predictions[rf_method], METHOD_CN[rf_method])

    mhd_matrix = confusion_matrix(
        labels, predictions[mhd_method], labels=GENRES, normalize="true"
    )
    rf_matrix = confusion_matrix(
        labels, predictions[rf_method], labels=GENRES, normalize="true"
    )
    recall_change = 100 * (np.diag(rf_matrix) - np.diag(mhd_matrix))
    genre_names = [GENRE_CN[genre] for genre in GENRES]
    colors = [TEAL if value >= 0 else CORAL for value in recall_change]
    bars = axes[2].barh(np.arange(len(GENRES)), recall_change, color=colors)
    axes[2].axvline(0, color=SLATE, linewidth=1)
    axes[2].set_yticks(np.arange(len(GENRES)), genre_names)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("召回率变化（百分点）")
    axes[2].set_title("随机森林相对 MHD")
    for bar, value in zip(bars, recall_change):
        axes[2].text(
            value + (0.8 if value >= 0 else -0.8),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.0f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            weight="bold",
        )
    clean_axis(axes[2], "x")
    fig.suptitle(
        "主要模型的折外混淆结构与类别改进",
        fontsize=16,
        weight="bold",
        color=NAVY,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.82, wspace=0.32)
    save_figure(fig, "confusion_models_cn.png")

    fusion_method = "MHD-RF probability fusion"
    fusion_fig, fusion_ax = plt.subplots(figsize=(6.2, 5.2))
    draw_confusion(
        fusion_ax,
        labels,
        predictions[fusion_method],
        METHOD_CN[fusion_method],
    )
    fusion_fig.tight_layout()
    save_figure(fusion_fig, "confusion_fusion_cn.png")


def plot_feature_importance(predictions: dict) -> None:
    importances = predictions["rf_importances"]
    names = predictions["feature_names"]
    top = np.argsort(importances)[-14:]
    values = importances[top]
    labels = [FEATURE_CN.get(str(name), str(name)) for name in names[top]]
    colors = [TEAL] * len(top)
    colors[-3:] = [GOLD, GOLD, NAVY]
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    bars = ax.barh(np.arange(len(top)), values, color=colors, height=0.68)
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel("打乱特征后折外平衡准确率的平均下降量")
    ax.set_title("单特征折外置换重要性（附录诊断）")
    for bar, value in zip(bars, values):
        ax.text(value + max(0.0005, values.max() * 0.025), bar.get_y() + 0.34, f"{value:.3f}", va="center", fontsize=8.5)
    clean_axis(ax, "x")
    fig.tight_layout()
    save_figure(fig, "feature_importance_cn.png")


def plot_group_importance() -> None:
    data = pd.read_csv(RAW_TABLES / "feature_group_importance.csv")
    group_cn = {
        "scale": "尺度与长度",
        "pitch": "音高统计",
        "interval_contour": "音程与轮廓",
        "dynamics": "力度动态",
        "rhythm": "节奏",
        "tonality": "音级与调性",
    }
    data["label"] = data["feature_group"].map(group_cn)
    data = data.sort_values("importance_mean")
    values = 100 * data["importance_mean"].to_numpy(float)
    low = values - 100 * data["importance_ci_low"].to_numpy(float)
    high = 100 * data["importance_ci_high"].to_numpy(float) - values
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    y = np.arange(len(data))
    ax.barh(y, values, color=TEAL, height=0.62)
    ax.errorbar(
        values,
        y,
        xerr=np.vstack((np.maximum(low, 0), np.maximum(high, 0))),
        fmt="none",
        ecolor=NAVY,
        capsize=3,
        linewidth=1.1,
    )
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.set_yticks(y, data["label"])
    ax.set_xlabel("联合打乱该特征组后的平衡准确率下降（百分点）")
    ax.set_title("随机森林的组级折外置换重要性")
    clean_axis(ax, "x")
    fig.tight_layout()
    save_figure(fig, "feature_group_importance_cn.png")


def plot_sensitivity() -> None:
    data = pd.read_csv(RAW_TABLES / "sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for ax, parameter, xlabel, title in (
        (axes[0], "resample_points", "重采样点数", "曲线离散密度"),
        (axes[1], "velocity_weight", "力度权重 $w_v$", "力度维权重"),
    ):
        selected = data[data["parameter"] == parameter]
        x = selected["value"].to_numpy(float)
        y = 100 * selected["fold_mean"].to_numpy(float)
        error = 100 * selected["fold_std"].to_numpy(float)
        ax.errorbar(x, y, yerr=error, marker="o", capsize=4, color=TEAL, linewidth=2)
        ax.axhline(20, color=CORAL, linestyle="--", linewidth=1.2)
        for x_value, y_value in zip(x, y):
            ax.annotate(f"{y_value:.1f}%", (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("分组五折准确率（%）")
        clean_axis(ax)
    fig.suptitle("MHD 模型的参数敏感性", fontsize=16, weight="bold", color=NAVY)
    fig.tight_layout()
    save_figure(fig, "sensitivity_cn.png")


def plot_repeated_validation() -> None:
    data = pd.read_csv(RAW_TABLES / "sample_split_validation.csv")
    methods = (
        "Tuned MHD (nested)",
        "RF descriptors",
        "MHD-RF probability fusion",
    )
    colors = (GOLD, NAVY, CORAL)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    sample_seeds = sorted(data["sample_seed"].unique())
    markers = ("o", "s", "^", "D")
    for index, (method, color) in enumerate(zip(methods, colors), start=1):
        selected_method = data[data["method"] == method]
        values = 100 * selected_method["accuracy"].to_numpy()
        for sample_index, sample_seed in enumerate(sample_seeds):
            selected = selected_method[selected_method["sample_seed"] == sample_seed]
            sample_values = 100 * selected["accuracy"].to_numpy()
            offsets = np.linspace(-0.13, 0.13, len(sample_values))
            ax.scatter(
                np.full(len(sample_values), index) + offsets,
                sample_values,
                s=46,
                marker=markers[sample_index % len(markers)],
                color=color,
                alpha=0.72,
                edgecolor="white",
                label=(
                    f"样本种子 {sample_seed}"
                    if index == 1
                    else None
                ),
            )
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0
        ax.errorbar(index, mean, yerr=std, fmt="D", color=color, capsize=5, linewidth=2)
        ax.text(index, mean + std + 1.2, f"{mean:.1f}% ± {std:.1f}%", ha="center", color=color, weight="bold")
    ax.axhline(20, color=CORAL, linestyle="--", linewidth=1.1)
    ax.set_xticks(range(1, 4), ["多参数 MHD", "随机森林", "概率融合"])
    ax.set_ylabel("每次完整五折的折外准确率（%）")
    ax.set_title("三个样本种子 × 五组外层划分下的模型稳定性")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, "repeated_validation_cn.png")


def plot_ablation() -> None:
    data = pd.read_csv(RAW_TABLES / "feature_ablation_repeated.csv")
    group_cn = {
        "scale": "尺度与长度",
        "pitch": "音高统计",
        "interval_contour": "音程与轮廓",
        "dynamics": "力度动态",
        "rhythm": "节奏",
        "tonality": "音级与调性",
    }
    data["label"] = data["removed_group"].map(group_cn)
    data = data.sort_values("balanced_accuracy_change_mean")
    change = 100 * data["balanced_accuracy_change_mean"].to_numpy()
    error = 100 * data["balanced_accuracy_change_std"].to_numpy()
    colors = [CORAL if value < 0 else TEAL for value in change]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(np.arange(len(data)), change, color=colors)
    ax.errorbar(
        change,
        np.arange(len(data)),
        xerr=error,
        fmt="none",
        ecolor=NAVY,
        capsize=3,
        linewidth=1.1,
    )
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.set_yticks(np.arange(len(data)), data["label"])
    ax.set_xlabel("移除该特征组后的平衡准确率变化（百分点）")
    ax.set_title("五组外层划分下的特征组消融（均值 ± 标准差）")
    for bar, value in zip(bars, change):
        ax.text(
            value + (0.18 if value >= 0 else -0.18),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
        )
    clean_axis(ax, "x")
    fig.tight_layout()
    save_figure(fig, "feature_ablation_cn.png")


def plot_fusion_complementarity(labels: np.ndarray, predictions: dict) -> None:
    mhd = predictions["Tuned MHD (nested)"] == labels
    rf = predictions["RF descriptors"] == labels
    fusion = predictions["MHD-RF probability fusion"] == labels
    categories = [
        ("两者均正确", mhd & rf),
        ("仅 MHD 正确", mhd & ~rf),
        ("仅随机森林正确", ~mhd & rf),
        ("两者均错误", ~mhd & ~rf),
    ]
    counts = np.asarray([mask.sum() for _, mask in categories])
    fusion_correct = np.asarray([(fusion & mask).sum() for _, mask in categories])
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(categories))
    ax.bar(x, counts, color="#D9E1EA", label="该分歧类别样本数")
    ax.bar(x, fusion_correct, color=CORAL, label="其中融合模型预测正确")
    ax.set_xticks(x, [label for label, _ in categories])
    ax.set_ylabel("样本数")
    ax.set_title("几何模型与描述符模型的互补性及融合收益")
    ax.legend(frameon=False)
    for index, (total, correct) in enumerate(zip(counts, fusion_correct)):
        ax.text(index, total + max(counts) * 0.025, f"{correct}/{total}", ha="center", fontsize=9)
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, "fusion_complementarity_cn.png")


def plot_dashboard(summary: dict) -> None:
    models = {row["method"]: row for row in summary["models"]}
    stats = summary["distance_statistics_mhd"]
    cards = [
        (f"{summary['n_samples']}", "五类平衡样本", NAVY),
        (f"{100 * models['Tuned MHD (nested)']['accuracy']:.2f}%", "主要几何模型", TEAL),
        (f"{100 * models['RF descriptors']['accuracy']:.2f}%", "随机森林描述符", GOLD),
        (f"{100 * models['MHD-RF probability fusion']['accuracy']:.2f}%", "概率融合", CORAL),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 3.25))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.25)
    ax.axis("off")
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
        ax.text(x + 0.28, 1.62, value, fontsize=23, weight="bold", color=color)
        ax.text(x + 0.28, 1.10, label, fontsize=10.5)
    ax.text(0.28, 2.86, "核心结果一览", fontsize=17, weight="bold", color=NAVY)
    ax.text(
        11.72,
        2.87,
        f"MHD 均值差 {stats['mean_gap']:.4f} · AUC {stats['pair_auc']:.3f} · "
        f"$p={stats['permutation_p']:.4f}$",
        fontsize=9.5,
        color=SLATE,
        ha="right",
    )
    save_figure(fig, "results_dashboard_cn.png")


def main() -> None:
    configure_style()
    summary, dataset, matrices, prediction_assets = load_assets()
    labels = dataset["labels"]
    stats = distance_arrays(summary, matrices["mhd_tpv"], labels)
    genre_matrix = np.loadtxt(
        RAW_TABLES / "genre_mean_distance.csv", delimiter=",", skiprows=1
    )
    primary_methods = (
        "Tuned MHD (nested)",
        "RF descriptors",
        "MHD-RF probability fusion",
    )
    primary_predictions = {
        method: prediction_assets[f"{prediction_key(method)}_predictions"]
        for method in primary_methods
    }

    export_tables(summary, dataset, prediction_assets)
    plot_data_funnel(summary)
    method_metadata = preserve_method_taxonomy()
    plot_dashboard(summary)
    plot_example_curves(dataset)
    plot_synthetic()
    plot_distance_distribution(stats)
    plot_heatmap(genre_matrix)
    plot_model_comparison(summary)
    plot_confusions(labels, primary_predictions)
    plot_mds(matrices["mhd_tpv"], labels)
    plot_feature_importance(prediction_assets)
    plot_group_importance()
    plot_sensitivity()
    plot_repeated_validation()
    plot_ablation()
    plot_fusion_complementarity(labels, primary_predictions)
    print(f"[paper-assets] preserved method figure: {method_metadata['sha256']}")
    print(f"[paper-assets] figures: {PAPER_FIGURES}")
    print(f"[paper-assets] tables: {PAPER_TABLES}")


if __name__ == "__main__":
    main()
