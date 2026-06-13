"""Reproducible final experiment for melody-curve genre classification.

Run from the repository root:
    D:/app/Anaconda/python.exe code/final_experiment.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from midi_geometry import (  # noqa: E402
    canonical_title,
    load_one_midi,
    local_minmax_curve,
    nearest_distance_summaries,
    relative_curve,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "adl-piano-midi"
RESULTS_DIR = ROOT / "results" / "final"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"
GENRES = ["Classical", "Jazz", "Rock", "Blues", "Electronic"]
COLORS = {
    "Classical": "#4472C4",
    "Jazz": "#ED7D31",
    "Rock": "#A5A5A5",
    "Blues": "#5B9BD5",
    "Electronic": "#70AD47",
}


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, FIGURE_DIR, TABLE_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_dataset(
    per_genre: int,
    n_points: int,
    seed: int,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    cache_path = CACHE_DIR / f"dataset_g{per_genre}_p{n_points}_s{seed}.npz"
    if cache_path.exists() and not rebuild:
        print(f"[data] loading cache: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}

    rng = np.random.default_rng(seed)
    all_titles: dict[str, set[str]] = {}
    files_by_genre: dict[str, list[Path]] = {}
    for genre in GENRES:
        files = sorted((DATA_DIR / genre).glob("*.mid"))
        files_by_genre[genre] = files
        for path in files:
            title = canonical_title(path)
            all_titles.setdefault(title, set()).add(genre)

    cross_genre_titles = {
        title for title, title_genres in all_titles.items() if len(title_genres) > 1
    }
    print(f"[data] excluding {len(cross_genre_titles)} cross-genre title groups")

    curves = []
    features = []
    labels = []
    groups = []
    filenames = []
    metadata_rows = []
    feature_names: list[str] | None = None

    for genre in GENRES:
        candidates = [
            path
            for path in files_by_genre[genre]
            if canonical_title(path) not in cross_genre_titles
        ]
        order = rng.permutation(len(candidates))
        failures = Counter()
        accepted = 0
        for index in order:
            path = candidates[int(index)]
            try:
                curve, feat, names, meta = load_one_midi(path, n_points=n_points)
            except Exception as exc:
                failures[type(exc).__name__] += 1
                continue
            curves.append(curve)
            features.append(feat)
            labels.append(genre)
            groups.append(canonical_title(path))
            filenames.append(path.name)
            feature_names = names
            metadata_rows.append(
                {
                    "genre": genre,
                    "file": path.name,
                    "group": canonical_title(path),
                    **meta,
                }
            )
            accepted += 1
            if accepted >= per_genre:
                break
        if accepted < per_genre:
            raise RuntimeError(
                f"{genre}: only {accepted}/{per_genre} usable files; failures={failures}"
            )
        print(f"[data] {genre}: {accepted} accepted, failures={sum(failures.values())}")

    result = {
        "curves": np.asarray(curves, dtype=np.float64),
        "features": np.asarray(features, dtype=np.float64),
        "labels": np.asarray(labels, dtype=str),
        "groups": np.asarray(groups, dtype=str),
        "filenames": np.asarray(filenames, dtype=str),
        "feature_names": np.asarray(feature_names, dtype=str),
    }
    np.savez_compressed(cache_path, **result)
    save_csv(TABLE_DIR / "sample_metadata.csv", metadata_rows)
    print(f"[data] cache written: {cache_path}")
    return result


def _fill_distance_row(
    i: int,
    curves: np.ndarray,
    trees: list[cKDTree],
    wanted: tuple[str, ...],
) -> tuple[int, dict[str, np.ndarray]]:
    n = len(curves)
    rows = {name: np.zeros(n, dtype=np.float64) for name in wanted}
    for j in range(i + 1, n):
        hd, q95, mhd = nearest_distance_summaries(
            curves[i], curves[j], trees[i], trees[j]
        )
        values = {"hd": hd, "q95": q95, "mhd": mhd}
        for name in wanted:
            rows[name][j] = values[name]
    return i, rows


def distance_family(
    curves: np.ndarray,
    wanted: tuple[str, ...],
    workers: int,
    label: str,
) -> dict[str, np.ndarray]:
    n = len(curves)
    matrices = {name: np.zeros((n, n), dtype=np.float64) for name in wanted}
    trees = [cKDTree(curve) for curve in curves]
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_fill_distance_row, i, curves, trees, wanted)
            for i in range(n - 1)
        ]
        for future in as_completed(futures):
            i, rows = future.result()
            for name, row in rows.items():
                matrices[name][i, i + 1 :] = row[i + 1 :]
                matrices[name][i + 1 :, i] = row[i + 1 :]
            completed += 1
            if completed % 40 == 0 or completed == n - 1:
                elapsed = time.time() - started
                print(f"[distance:{label}] rows {completed}/{n-1}, {elapsed:.1f}s")
    return matrices


def dtw_pair(a: np.ndarray, b: np.ndarray, window: int = 14) -> float:
    n, m = len(a), len(b)
    window = max(window, abs(n - m))
    previous = np.full(m + 1, np.inf)
    current = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current.fill(np.inf)
        start = max(1, i - window)
        stop = min(m, i + window)
        for j in range(start, stop + 1):
            cost = abs(a[i - 1] - b[j - 1])
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    return float(previous[m] / (n + m))


def dtw_matrix(curves: np.ndarray, workers: int) -> np.ndarray:
    pitch = np.asarray(
        [(curve[:, 1] - np.median(curve[:, 1])) / 12.0 for curve in curves]
    )
    n = len(pitch)
    matrix = np.zeros((n, n), dtype=np.float64)

    def row_task(i: int) -> tuple[int, np.ndarray]:
        row = np.zeros(n)
        for j in range(i + 1, n):
            row[j] = dtw_pair(pitch[i], pitch[j])
        return i, row

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(row_task, i) for i in range(n - 1)]
        for completed, future in enumerate(as_completed(futures), start=1):
            i, row = future.result()
            matrix[i, i + 1 :] = row[i + 1 :]
            matrix[i + 1 :, i] = row[i + 1 :]
            if completed % 40 == 0 or completed == n - 1:
                print(
                    f"[distance:dtw] rows {completed}/{n-1}, "
                    f"{time.time()-started:.1f}s"
                )
    return matrix


def validate_distance_matrix(name: str, matrix: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name}: distance matrix is not square")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError(f"{name}: distance matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12):
        raise ValueError(f"{name}: distance matrix diagonal is not zero")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError(f"{name}: distance matrix contains invalid values")
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    nonzero_fraction = float(np.mean(upper > 1e-12))
    if nonzero_fraction < 0.95:
        raise ValueError(
            f"{name}: only {nonzero_fraction:.1%} pairwise distances are nonzero"
        )


def compute_distances(
    base_curves: np.ndarray,
    per_genre: int,
    n_points: int,
    seed: int,
    workers: int,
    rebuild: bool = False,
) -> dict[str, np.ndarray]:
    cache_path = CACHE_DIR / f"distances_g{per_genre}_p{n_points}_s{seed}.npz"
    if cache_path.exists() and not rebuild:
        print(f"[distance] loading cache: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=False)
        cached = {key: loaded[key] for key in loaded.files}
        for name, matrix in cached.items():
            validate_distance_matrix(name, matrix)
        return cached

    local = np.asarray([local_minmax_curve(curve) for curve in base_curves])
    tp = np.asarray([relative_curve(curve, velocity_weight=0.0) for curve in base_curves])
    tpv = np.asarray(
        [relative_curve(curve, velocity_weight=0.25) for curve in base_curves]
    )

    result = {}
    result["hd_local"] = distance_family(
        local, ("hd",), workers, "hd-local"
    )["hd"]
    result["hd_tp"] = distance_family(tp, ("hd",), workers, "hd-tp")["hd"]
    robust = distance_family(tpv, ("hd", "q95", "mhd"), workers, "tpv")
    result["hd_tpv"] = robust["hd"]
    result["q95_tpv"] = robust["q95"]
    result["mhd_tpv"] = robust["mhd"]
    result["dtw_pitch"] = dtw_matrix(base_curves, workers)
    for name, matrix in result.items():
        validate_distance_matrix(name, matrix)
    np.savez_compressed(cache_path, **result)
    print(f"[distance] cache written: {cache_path}")
    return result


def predict_knn(
    distance_matrix: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    predictions = []
    for test in test_idx:
        ordered = train_idx[np.argsort(distance_matrix[test, train_idx])[:k]]
        neighbor_labels = labels[ordered]
        counts = Counter(neighbor_labels)
        max_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == max_count}
        predictions.append(next(label for label in neighbor_labels if label in tied))
    return np.asarray(predictions)


def choose_k(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: tuple[int, ...] = (1, 3, 5, 7, 9),
) -> int:
    inner_labels = labels[train_idx]
    inner_groups = groups[train_idx]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)
    scores = {k: [] for k in candidates}
    for inner_train, inner_test in splitter.split(
        np.zeros(len(train_idx)), inner_labels, inner_groups
    ):
        global_train = train_idx[inner_train]
        global_test = train_idx[inner_test]
        for k in candidates:
            pred = predict_knn(matrix, global_train, global_test, labels, k)
            scores[k].append(balanced_accuracy_score(labels[global_test], pred))
    return max(candidates, key=lambda k: (np.mean(scores[k]), -k))


def evaluate_distance_metric(
    name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.empty(len(labels), dtype=labels.dtype)
    fold_rows = []
    selected_k = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        k = choose_k(matrix, train_idx, labels, groups)
        selected_k.append(k)
        pred = predict_knn(matrix, train_idx, test_idx, labels, k)
        predictions[test_idx] = pred
        fold_rows.append(
            {
                "method": name,
                "fold": fold,
                "k": k,
                "n_test": len(test_idx),
                "accuracy": accuracy_score(labels[test_idx], pred),
                "balanced_accuracy": balanced_accuracy_score(labels[test_idx], pred),
                "macro_f1": f1_score(labels[test_idx], pred, average="macro"),
            }
        )

    fold_acc = np.asarray([row["accuracy"] for row in fold_rows])
    summary = {
        "method": name,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "fold_mean": fold_acc.mean(),
        "fold_std": fold_acc.std(ddof=1),
        "k_mode": Counter(selected_k).most_common(1)[0][0],
        "k_selected": "/".join(map(str, selected_k)),
    }
    return summary, predictions, np.asarray(fold_rows, dtype=object)


def evaluate_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict, np.ndarray, list[dict], np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.empty(len(labels), dtype=labels.dtype)
    fold_rows = []
    importances = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=100 + fold,
            n_jobs=-1,
        )
        model.fit(features[train_idx], labels[train_idx])
        pred = model.predict(features[test_idx])
        predictions[test_idx] = pred
        importances.append(model.feature_importances_)
        fold_rows.append(
            {
                "method": "RF descriptors",
                "fold": fold,
                "k": "",
                "n_test": len(test_idx),
                "accuracy": accuracy_score(labels[test_idx], pred),
                "balanced_accuracy": balanced_accuracy_score(labels[test_idx], pred),
                "macro_f1": f1_score(labels[test_idx], pred, average="macro"),
            }
        )
    fold_acc = np.asarray([row["accuracy"] for row in fold_rows])
    summary = {
        "method": "RF descriptors",
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "fold_mean": fold_acc.mean(),
        "fold_std": fold_acc.std(ddof=1),
        "k_mode": "",
        "k_selected": "",
    }
    return summary, predictions, fold_rows, np.mean(importances, axis=0)


def evaluate_weighted_mhd(
    velocity_matrices: dict[float, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict, np.ndarray, list[dict]]:
    candidates = (1, 3, 5, 7, 9)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.empty(len(labels), dtype=labels.dtype)
    fold_rows = []
    selections = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        inner_labels = labels[train_idx]
        inner_groups = groups[train_idx]
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026)
        scores = {
            (weight, k): []
            for weight in velocity_matrices
            for k in candidates
        }
        for inner_train, inner_test in inner.split(
            np.zeros(len(train_idx)), inner_labels, inner_groups
        ):
            global_train = train_idx[inner_train]
            global_test = train_idx[inner_test]
            for weight, matrix in velocity_matrices.items():
                for k in candidates:
                    pred = predict_knn(
                        matrix, global_train, global_test, labels, k
                    )
                    scores[(weight, k)].append(
                        balanced_accuracy_score(labels[global_test], pred)
                    )
        weight, k = max(
            scores,
            key=lambda item: (np.mean(scores[item]), -item[0], -item[1]),
        )
        selections.append((weight, k))
        pred = predict_knn(
            velocity_matrices[weight], train_idx, test_idx, labels, k
        )
        predictions[test_idx] = pred
        fold_rows.append(
            {
                "method": "Weighted MHD (nested)",
                "fold": fold,
                "k": k,
                "n_test": len(test_idx),
                "accuracy": accuracy_score(labels[test_idx], pred),
                "balanced_accuracy": balanced_accuracy_score(labels[test_idx], pred),
                "macro_f1": f1_score(labels[test_idx], pred, average="macro"),
                "velocity_weight": weight,
            }
        )

    fold_acc = np.asarray([row["accuracy"] for row in fold_rows])
    summary = {
        "method": "Weighted MHD (nested)",
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "fold_mean": fold_acc.mean(),
        "fold_std": fold_acc.std(ddof=1),
        "k_mode": Counter(k for _, k in selections).most_common(1)[0][0],
        "k_selected": "/".join(str(k) for _, k in selections),
        "weight_selected": "/".join(f"{weight:.2f}" for weight, _ in selections),
    }
    return summary, predictions, fold_rows


def distance_statistics(
    matrix: np.ndarray,
    labels: np.ndarray,
    seed: int,
    permutations: int = 499,
) -> dict:
    upper = np.triu_indices(len(labels), k=1)
    distances = matrix[upper]
    same = labels[upper[0]] == labels[upper[1]]
    within = distances[same]
    between = distances[~same]
    auc = roc_auc_score(same.astype(int), -distances)
    observed_gap = float(between.mean() - within.mean())

    rng = np.random.default_rng(seed)
    perm_gaps = np.empty(permutations)
    for i in range(permutations):
        shuffled = rng.permutation(labels)
        perm_same = shuffled[upper[0]] == shuffled[upper[1]]
        perm_gaps[i] = (
            distances[~perm_same].mean() - distances[perm_same].mean()
        )
    p_value = (1 + np.sum(perm_gaps >= observed_gap)) / (permutations + 1)
    return {
        "within_mean": float(within.mean()),
        "within_median": float(np.median(within)),
        "between_mean": float(between.mean()),
        "between_median": float(np.median(between)),
        "mean_gap": observed_gap,
        "pair_auc": float(auc),
        "permutation_p": float(p_value),
        "within": within,
        "between": between,
    }


def resample_curve_batch(curves: np.ndarray, n_points: int) -> np.ndarray:
    old_grid = curves[0, :, 0]
    new_grid = np.linspace(0.0, 1.0, n_points)
    result = np.empty((len(curves), n_points, 3), dtype=np.float64)
    result[:, :, 0] = new_grid
    for i, curve in enumerate(curves):
        result[i, :, 1] = np.interp(new_grid, old_grid, curve[:, 1])
        result[i, :, 2] = np.interp(new_grid, old_grid, curve[:, 2])
    return result


def run_sensitivity(
    base_curves: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    existing_mhd: np.ndarray,
    per_genre: int,
    seed: int,
    workers: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    cache_path = CACHE_DIR / f"sensitivity_mhd_g{per_genre}_s{seed}.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        cached = {key: loaded[key] for key in loaded.files}
        for name, matrix in cached.items():
            validate_distance_matrix(name, matrix)

    matrices: dict[str, np.ndarray] = {"points_96": existing_mhd, "velocity_0.25": existing_mhd}
    point_counts = (48, 72, 96)
    velocity_weights = (0.0, 0.10, 0.25, 0.50)

    for n_points in point_counts:
        key = f"points_{n_points}"
        if key in matrices:
            continue
        if key in cached:
            matrices[key] = cached[key]
            continue
        sampled = resample_curve_batch(base_curves, n_points)
        represented = np.asarray([relative_curve(curve, 0.25) for curve in sampled])
        matrices[key] = distance_family(
            represented, ("mhd",), workers, key
        )["mhd"]
        validate_distance_matrix(key, matrices[key])

    for weight in velocity_weights:
        key = f"velocity_{weight:.2f}"
        if weight == 0.25:
            matrices[key] = existing_mhd
            continue
        if key in cached:
            matrices[key] = cached[key]
            continue
        represented = np.asarray([relative_curve(curve, weight) for curve in base_curves])
        matrices[key] = distance_family(
            represented, ("mhd",), workers, key
        )["mhd"]
        validate_distance_matrix(key, matrices[key])

    np.savez_compressed(cache_path, **matrices)
    rows = []
    for n_points in point_counts:
        summary, _, _ = evaluate_distance_metric(
            f"MHD points={n_points}", matrices[f"points_{n_points}"], labels, groups
        )
        rows.append(
            {
                "parameter": "resample_points",
                "value": n_points,
                "fold_mean": summary["fold_mean"],
                "fold_std": summary["fold_std"],
            }
        )
    for weight in velocity_weights:
        summary, _, _ = evaluate_distance_metric(
            f"MHD velocity={weight:.2f}",
            matrices[f"velocity_{weight:.2f}"],
            labels,
            groups,
        )
        rows.append(
            {
                "parameter": "velocity_weight",
                "value": weight,
                "fold_mean": summary["fold_mean"],
                "fold_std": summary["fold_std"],
            }
        )
    return rows, matrices


def genre_mean_matrix(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(GENRES), len(GENRES)))
    for i, genre_a in enumerate(GENRES):
        idx_a = np.flatnonzero(labels == genre_a)
        for j, genre_b in enumerate(GENRES):
            idx_b = np.flatnonzero(labels == genre_b)
            block = matrix[np.ix_(idx_a, idx_b)]
            if i == j:
                block = block[np.triu_indices_from(block, k=1)]
            result[i, j] = block.mean()
    return result


def synthetic_robustness() -> list[dict]:
    t = np.linspace(0, 1, 96)
    base = np.column_stack((t, 0.55 * np.sin(4 * np.pi * t), np.zeros_like(t)))
    rows = []
    for amplitude in np.linspace(0, 2.0, 11):
        perturbed = base.copy()
        perturbed[48, 1] += amplitude
        hd, q95, mhd = nearest_distance_summaries(base, perturbed)
        rows.append(
            {
                "outlier_amplitude": amplitude,
                "hd": hd,
                "q95": q95,
                "mhd": mhd,
            }
        )
    return rows


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    labels = [
        "MIDI",
        "Skyline\nmelody",
        "Time-grid\nresampling",
        "Scaled 3D\ncurve",
        "Hausdorff\nfamily",
        "Grouped\nvalidation",
    ]
    xs = np.linspace(0.07, 0.93, len(labels))
    for x, label in zip(xs, labels):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#EAF2F8", ec="#2E75B6"),
        )
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(right - 0.055, 0.5),
            xytext=(left + 0.055, 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"),
        )
    fig.savefig(FIGURE_DIR / "workflow.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_example_curves(curves: np.ndarray, labels: np.ndarray) -> None:
    fig = plt.figure(figsize=(13, 8))
    for index, genre in enumerate(GENRES, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        curve = relative_curve(curves[np.flatnonzero(labels == genre)[0]], 0.25)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=COLORS[genre], lw=1.5)
        ax.scatter(curve[::6, 0], curve[::6, 1], curve[::6, 2], s=8)
        ax.set_title(genre)
        ax.set_xlabel("phase")
        ax.set_ylabel("relative pitch")
        ax.set_zlabel("velocity")
    fig.suptitle("Representative normalized melody curves")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "example_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_distribution(stats_row: dict) -> None:
    within = stats_row["within"]
    between = stats_row["between"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(
        min(within.min(), between.min()), max(within.max(), between.max()), 45
    )
    axes[0].hist(within, bins=bins, density=True, alpha=0.65, label="within")
    axes[0].hist(between, bins=bins, density=True, alpha=0.65, label="between")
    axes[0].set_xlabel("Modified Hausdorff distance")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].boxplot([within, between], tick_labels=["within", "between"], showfliers=False)
    axes[1].set_ylabel("Modified Hausdorff distance")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "distance_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="YlGnBu")
    ax.set_xticks(range(len(GENRES)), GENRES, rotation=35, ha="right")
    ax.set_yticks(range(len(GENRES)), GENRES)
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="mean Modified Hausdorff distance")
    ax.set_title("Genre-level mean distance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "genre_distance_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mds(matrix: np.ndarray, labels: np.ndarray) -> None:
    embedding = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=2,
        max_iter=300,
    ).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    for genre in GENRES:
        mask = labels == genre
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=25,
            alpha=0.7,
            label=genre,
            color=COLORS[genre],
        )
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("MDS projection of Modified Hausdorff distances")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mds_mhd.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summaries: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [row["method"] for row in summaries]
    means = [row["fold_mean"] for row in summaries]
    stds = [row["fold_std"] for row in summaries]
    colors = ["#A5A5A5", "#5B9BD5", "#4472C4", "#ED7D31", "#70AD47", "#7030A0", "#C55A11"]
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors[: len(names)])
    ax.axhline(0.2, color="black", linestyle="--", lw=1, label="random baseline")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_ylabel("5-fold grouped-CV accuracy")
    ax.set_ylim(0, max(means) + max(stds) + 0.12)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(labels: np.ndarray, predictions: np.ndarray, filename: str) -> None:
    cm = confusion_matrix(labels, predictions, labels=GENRES, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(0.5, cm.max()))
    ax.set_xticks(range(len(GENRES)), GENRES, rotation=35, ha="right")
    ax.set_yticks(range(len(GENRES)), GENRES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            ax.text(j, i, f"{cm[i, j]*100:.0f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [row["outlier_amplitude"] for row in rows]
    for key, label in [("hd", "max-HD"), ("q95", "Q95-HD"), ("mhd", "MHD")]:
        ax.plot(x, [row[key] for row in rows], marker="o", label=label)
    ax.set_xlabel("single-point perturbation amplitude")
    ax.set_ylabel("distance from original curve")
    ax.set_title("Robustness to one outlying note")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "synthetic_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, names: np.ndarray) -> None:
    top = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), importances[top], color="#70AD47")
    ax.set_yticks(range(len(top)), names[top])
    ax.set_xlabel("mean feature importance")
    ax.set_title("Random-forest descriptor importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, parameter, title, xlabel in [
        (axes[0], "resample_points", "Resampling sensitivity", "number of points"),
        (axes[1], "velocity_weight", "Velocity-weight sensitivity", "velocity weight"),
    ]:
        selected = [row for row in rows if row["parameter"] == parameter]
        x = np.asarray([float(row["value"]) for row in selected])
        y = np.asarray([float(row["fold_mean"]) for row in selected])
        error = np.asarray([float(row["fold_std"]) for row in selected])
        ax.errorbar(x, y, yerr=error, marker="o", capsize=4, color="#4472C4")
        ax.axhline(0.2, color="black", linestyle="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("grouped-CV accuracy")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-genre", type=int, default=80)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=max(2, min(8, os.cpu_count() or 2)))
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--rebuild-distances", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    print(f"[config] genres={GENRES}, per_genre={args.per_genre}, points={args.points}")
    dataset = sample_dataset(
        args.per_genre, args.points, args.seed, rebuild=args.rebuild_data
    )
    curves = dataset["curves"]
    labels = dataset["labels"]
    groups = dataset["groups"]
    features = dataset["features"]

    matrices = compute_distances(
        curves,
        args.per_genre,
        args.points,
        args.seed,
        args.workers,
        rebuild=args.rebuild_distances,
    )

    method_names = {
        "hd_local": "HD local min-max",
        "hd_tp": "HD relative TP",
        "hd_tpv": "HD relative TPV",
        "q95_tpv": "Q95-HD relative TPV",
        "mhd_tpv": "MHD relative TPV",
        "dtw_pitch": "DTW relative pitch",
    }

    summaries = []
    fold_rows: list[dict] = []
    predictions_by_method = {}
    for key in method_names:
        summary, predictions, folds = evaluate_distance_metric(
            method_names[key], matrices[key], labels, groups
        )
        summaries.append(summary)
        predictions_by_method[key] = predictions
        fold_rows.extend(folds.tolist())
        print(
            f"[eval] {summary['method']}: {summary['fold_mean']:.4f} "
            f"+/- {summary['fold_std']:.4f}, k={summary['k_selected']}"
        )

    stats_row = distance_statistics(matrices["mhd_tpv"], labels, args.seed)
    genre_matrix = genre_mean_matrix(matrices["mhd_tpv"], labels)
    synthetic_rows = synthetic_robustness()
    sensitivity_rows, sensitivity_matrices = run_sensitivity(
        curves,
        labels,
        groups,
        matrices["mhd_tpv"],
        args.per_genre,
        args.seed,
        args.workers,
    )
    velocity_matrices = {
        weight: sensitivity_matrices[f"velocity_{weight:.2f}"]
        for weight in (0.0, 0.10, 0.25, 0.50)
    }
    weighted_summary, weighted_predictions, weighted_folds = evaluate_weighted_mhd(
        velocity_matrices, labels, groups
    )
    summaries.append(weighted_summary)
    fold_rows.extend(weighted_folds)
    predictions_by_method["weighted_mhd"] = weighted_predictions
    method_names["weighted_mhd"] = "Weighted MHD (nested)"
    print(
        f"[eval] Weighted MHD (nested): {weighted_summary['fold_mean']:.4f} "
        f"+/- {weighted_summary['fold_std']:.4f}, "
        f"weights={weighted_summary['weight_selected']}"
    )

    rf_summary, rf_predictions, rf_folds, importances = evaluate_random_forest(
        features, labels, groups
    )
    summaries.append(rf_summary)
    fold_rows.extend(rf_folds)
    predictions_by_method["rf"] = rf_predictions
    print(
        f"[eval] RF descriptors: {rf_summary['fold_mean']:.4f} "
        f"+/- {rf_summary['fold_std']:.4f}"
    )

    save_csv(TABLE_DIR / "model_summary.csv", summaries)
    save_csv(TABLE_DIR / "fold_results.csv", fold_rows)
    save_csv(
        TABLE_DIR / "distance_statistics.csv",
        [
            {
                key: value
                for key, value in stats_row.items()
                if key not in ("within", "between")
            }
        ],
    )
    save_csv(TABLE_DIR / "synthetic_robustness.csv", synthetic_rows)
    save_csv(TABLE_DIR / "sensitivity.csv", sensitivity_rows)
    np.savetxt(
        TABLE_DIR / "genre_mean_distance.csv",
        genre_matrix,
        delimiter=",",
        header=",".join(GENRES),
        comments="",
    )

    best_hd_key = max(
        ("hd_local", "hd_tp", "hd_tpv", "q95_tpv", "mhd_tpv", "weighted_mhd"),
        key=lambda key: next(
            row["fold_mean"] for row in summaries if row["method"] == method_names[key]
        ),
    )
    run_summary = {
        "config": vars(args),
        "genres": GENRES,
        "n_samples": int(len(labels)),
        "class_counts": {genre: int(np.sum(labels == genre)) for genre in GENRES},
        "unique_groups": int(len(np.unique(groups))),
        "best_hausdorff_method": method_names[best_hd_key],
        "distance_statistics_mhd": {
            key: value
            for key, value in stats_row.items()
            if key not in ("within", "between")
        },
        "models": summaries,
        "sensitivity": sensitivity_rows,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_workflow()
    plot_example_curves(curves, labels)
    plot_distance_distribution(stats_row)
    plot_heatmap(genre_matrix)
    plot_mds(matrices["mhd_tpv"], labels)
    plot_model_comparison(summaries)
    plot_confusion(labels, predictions_by_method[best_hd_key], "confusion_best_hd.png")
    plot_confusion(labels, rf_predictions, "confusion_rf.png")
    plot_synthetic(synthetic_rows)
    plot_feature_importance(importances, dataset["feature_names"])
    plot_sensitivity(sensitivity_rows)

    print(f"[done] best Hausdorff variant: {method_names[best_hd_key]}")
    print(f"[done] results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
