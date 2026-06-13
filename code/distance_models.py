"""Distance construction and caching for melody-curve experiments."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

from data_pipeline import CACHE_DIR, DATA_DIR, PIPELINE_VERSION
from midi_geometry import (
    load_one_midi,
    local_minmax_curve,
    nearest_distance_summaries,
    relative_curve,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in minimal environments
    njit = None


def distance_cache_path(per_genre: int, n_points: int, seed: int) -> Path:
    return CACHE_DIR / (
        f"distances_{PIPELINE_VERSION}_g{per_genre}_p{n_points}_s{seed}.npz"
    )


def sensitivity_cache_path(per_genre: int, seed: int) -> Path:
    return CACHE_DIR / (
        f"sensitivity_mhd_{PIPELINE_VERSION}_g{per_genre}_s{seed}.npz"
    )


def dtw_cache_path(per_genre: int, seed: int) -> Path:
    return CACHE_DIR / f"dtw_grid_{PIPELINE_VERSION}_g{per_genre}_s{seed}.npz"


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
    if len(upper) and float(np.mean(upper > 1e-12)) < 0.95:
        raise ValueError(f"{name}: too many pairwise distances are zero")


def _fill_distance_row(
    index: int,
    curves: np.ndarray,
    trees: list[cKDTree],
    wanted: tuple[str, ...],
) -> tuple[int, dict[str, np.ndarray]]:
    rows = {name: np.zeros(len(curves), dtype=np.float64) for name in wanted}
    for other in range(index + 1, len(curves)):
        hd, q95, mhd = nearest_distance_summaries(
            curves[index], curves[other], trees[index], trees[other]
        )
        values = {"hd": hd, "q95": q95, "mhd": mhd}
        for name in wanted:
            rows[name][other] = values[name]
    return index, rows


def distance_family(
    curves: np.ndarray,
    wanted: tuple[str, ...],
    workers: int,
    label: str,
) -> dict[str, np.ndarray]:
    matrices = {
        name: np.zeros((len(curves), len(curves)), dtype=np.float64)
        for name in wanted
    }
    trees = [cKDTree(curve) for curve in curves]
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_fill_distance_row, index, curves, trees, wanted)
            for index in range(len(curves) - 1)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, rows = future.result()
            for name, row in rows.items():
                matrices[name][index, index + 1 :] = row[index + 1 :]
                matrices[name][index + 1 :, index] = row[index + 1 :]
            if completed % 50 == 0 or completed == len(curves) - 1:
                print(
                    f"[distance:{label}] rows {completed}/{len(curves)-1}, "
                    f"{time.time() - started:.1f}s"
                )
    return matrices


def phase_aligned_rmse_matrix(curves: np.ndarray, velocity_weight: float) -> np.ndarray:
    represented = np.asarray(
        [relative_curve(curve, velocity_weight)[:, 1:] for curve in curves]
    )
    flattened = represented.reshape(len(represented), -1)
    matrix = squareform(pdist(flattened, metric="euclidean"))
    matrix /= np.sqrt(represented.shape[1])
    return matrix


def compute_base_distances(
    curves: np.ndarray,
    per_genre: int,
    n_points: int,
    seed: int,
    workers: int,
    force: bool = False,
) -> dict[str, np.ndarray]:
    cache_path = distance_cache_path(per_genre, n_points, seed)
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=False)
        cached = {key: loaded[key] for key in loaded.files}
        if all(matrix.shape == (len(curves), len(curves)) for matrix in cached.values()):
            for name, matrix in cached.items():
                validate_distance_matrix(name, matrix)
            print(f"[distance] loading base matrices: {cache_path}")
            return cached

    local = np.asarray([local_minmax_curve(curve) for curve in curves])
    tp = np.asarray([relative_curve(curve, 0.0) for curve in curves])
    tpv = np.asarray([relative_curve(curve, 0.25) for curve in curves])
    result = {
        "hd_local": distance_family(local, ("hd",), workers, "hd-local")["hd"],
        "hd_tp": distance_family(tp, ("hd",), workers, "hd-tp")["hd"],
    }
    robust = distance_family(tpv, ("hd", "q95", "mhd"), workers, "tpv")
    result.update(
        {
            "hd_tpv": robust["hd"],
            "q95_tpv": robust["q95"],
            "mhd_tpv": robust["mhd"],
            "phase_rmse": phase_aligned_rmse_matrix(curves, 0.25),
        }
    )
    for name, matrix in result.items():
        validate_distance_matrix(name, matrix)
    np.savez_compressed(cache_path, **result)
    print(f"[distance] base cache written: {cache_path}")
    return result


def reload_curves(
    labels: np.ndarray,
    filenames: np.ndarray,
    n_points: int,
    workers: int,
) -> np.ndarray:
    def load_index(index: int) -> tuple[int, np.ndarray]:
        path = DATA_DIR / str(labels[index]) / str(filenames[index])
        curve, _, _, _ = load_one_midi(path, n_points=n_points)
        return index, curve

    curves = np.empty((len(labels), n_points, 3), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(load_index, index) for index in range(len(labels))]
        for future in as_completed(futures):
            index, curve = future.result()
            curves[index] = curve
    return curves


def build_mhd_grid(
    base_curves: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
    base_mhd: np.ndarray,
    per_genre: int,
    seed: int,
    workers: int,
    point_counts: tuple[int, ...] = (48, 72, 96),
    velocity_weights: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50),
    force: bool = False,
) -> dict[tuple[int, float], np.ndarray]:
    cache_path = sensitivity_cache_path(per_genre, seed)
    cached = {}
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=False)
        cached = {key: loaded[key] for key in loaded.files}

    curve_batches = {base_curves.shape[1]: base_curves}
    matrices: dict[tuple[int, float], np.ndarray] = {}
    serialized: dict[str, np.ndarray] = {}
    for n_points in point_counts:
        for weight in velocity_weights:
            key = f"points_{n_points}_velocity_{weight:.2f}"
            if n_points == base_curves.shape[1] and np.isclose(weight, 0.25):
                matrix = base_mhd
            elif key in cached and cached[key].shape == (len(labels), len(labels)):
                matrix = cached[key]
            else:
                if n_points not in curve_batches:
                    curve_batches[n_points] = reload_curves(
                        labels, filenames, n_points, workers
                    )
                represented = np.asarray(
                    [relative_curve(curve, weight) for curve in curve_batches[n_points]]
                )
                matrix = distance_family(
                    represented, ("mhd",), workers, f"mhd-{key}"
                )["mhd"]
            validate_distance_matrix(key, matrix)
            matrices[(n_points, weight)] = matrix
            serialized[key] = matrix
    np.savez_compressed(cache_path, **serialized)
    return matrices


def _dtw_python(a: np.ndarray, b: np.ndarray, window: int) -> float:
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
            cost = float(np.sqrt(np.sum((a[i - 1] - b[j - 1]) ** 2)))
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    return float(previous[m] / (n + m))


if njit is not None:
    _dtw_core = njit(cache=True, nogil=True)(_dtw_python)
else:  # pragma: no cover
    _dtw_core = _dtw_python


def multivariate_dtw_matrix(
    curves: np.ndarray,
    velocity_weight: float,
    window_fraction: float,
    workers: int,
    label: str,
) -> np.ndarray:
    sequences = np.asarray(
        [relative_curve(curve, velocity_weight)[:, 1:] for curve in curves],
        dtype=np.float64,
    )
    window = max(1, int(round(curves.shape[1] * window_fraction)))
    # Trigger compilation before worker threads start.
    _dtw_core(sequences[0], sequences[0], window)
    matrix = np.zeros((len(sequences), len(sequences)), dtype=np.float64)

    def row_task(index: int) -> tuple[int, np.ndarray]:
        row = np.zeros(len(sequences), dtype=np.float64)
        for other in range(index + 1, len(sequences)):
            row[other] = _dtw_core(sequences[index], sequences[other], window)
        return index, row

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(row_task, index) for index in range(len(sequences) - 1)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, row = future.result()
            matrix[index, index + 1 :] = row[index + 1 :]
            matrix[index + 1 :, index] = row[index + 1 :]
            if completed % 50 == 0 or completed == len(sequences) - 1:
                print(
                    f"[distance:{label}] rows {completed}/{len(sequences)-1}, "
                    f"{time.time() - started:.1f}s"
                )
    return matrix


def build_dtw_grid(
    labels: np.ndarray,
    filenames: np.ndarray,
    per_genre: int,
    seed: int,
    workers: int,
    point_counts: tuple[int, ...] = (36, 48),
    velocity_weights: tuple[float, ...] = (0.0, 0.25, 0.50),
    window_fractions: tuple[float, ...] = (0.10, 0.20),
    force: bool = False,
) -> dict[tuple[int, float, float], np.ndarray]:
    cache_path = dtw_cache_path(per_genre, seed)
    cached = {}
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=False)
        cached = {key: loaded[key] for key in loaded.files}

    curve_batches: dict[int, np.ndarray] = {}
    matrices: dict[tuple[int, float, float], np.ndarray] = {}
    serialized: dict[str, np.ndarray] = {}
    for n_points in point_counts:
        curve_batches[n_points] = reload_curves(labels, filenames, n_points, workers)
        for weight in velocity_weights:
            for window in window_fractions:
                key = f"points_{n_points}_velocity_{weight:.2f}_window_{window:.2f}"
                if key in cached and cached[key].shape == (len(labels), len(labels)):
                    matrix = cached[key]
                else:
                    matrix = multivariate_dtw_matrix(
                        curve_batches[n_points],
                        weight,
                        window,
                        workers,
                        key,
                    )
                validate_distance_matrix(key, matrix)
                matrices[(n_points, weight, window)] = matrix
                serialized[key] = matrix
                np.savez_compressed(cache_path, **serialized)
    return matrices
