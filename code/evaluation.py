"""Leakage-safe nested evaluation, uncertainty, and model comparison."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import GENRES
from midi_geometry import feature_group_indices


INNER_SEEDS = (2026, 2027, 2028)
REPEAT_OUTER_SEEDS = (11, 23, 42, 67, 101)
K_CANDIDATES = (1, 3, 5, 7, 9)
CLASS_ARRAY = np.asarray(GENRES)


def assert_group_separation(
    train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray
) -> None:
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise RuntimeError(f"group leakage detected: {sorted(overlap)[:3]}")


def _align_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(GENRES)), dtype=np.float64)
    for column, label in enumerate(classes):
        aligned[:, GENRES.index(str(label))] = probabilities[:, column]
    return aligned


def predict_knn_proba(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    probabilities = np.zeros((len(test_idx), len(GENRES)), dtype=np.float64)
    k = min(k, len(train_idx))
    for row, test in enumerate(test_idx):
        ordered = train_idx[
            np.argsort(matrix[test, train_idx], kind="stable")[:k]
        ]
        for rank, neighbor in enumerate(ordered):
            class_index = GENRES.index(str(labels[neighbor]))
            probabilities[row, class_index] += 1.0
            probabilities[row, class_index] += 1e-9 * (k - rank)
        probabilities[row] /= probabilities[row].sum()
    return probabilities


def predictions_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return CLASS_ARRAY[np.argmax(probabilities, axis=1)]


def metric_values(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def bootstrap_metric_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, float]:
    if iterations <= 0:
        return {}
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == genre) for genre in GENRES]
    samples = {name: np.empty(iterations) for name in metric_values(labels, predictions)}
    for iteration in range(iterations):
        bootstrap_idx = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        values = metric_values(labels[bootstrap_idx], predictions[bootstrap_idx])
        for name, value in values.items():
            samples[name][iteration] = value
    result = {}
    for name, values in samples.items():
        result[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return result


def summarize_predictions(
    method: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    fold_rows: list[dict],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    extra: dict | None = None,
) -> dict:
    fold_accuracy = np.asarray([row["accuracy"] for row in fold_rows])
    summary = {
        "method": method,
        **metric_values(labels, predictions),
        "fold_mean": float(fold_accuracy.mean()),
        "fold_std": float(fold_accuracy.std(ddof=1)),
        **bootstrap_metric_intervals(
            labels,
            predictions,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
    }
    if extra:
        summary.update(extra)
    return summary


def choose_k(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: tuple[int, ...] = K_CANDIDATES,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
) -> int:
    scores = {k: [] for k in candidates}
    for inner_seed in inner_seeds:
        splitter = StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=inner_seed
        )
        for inner_train, inner_test in splitter.split(
            np.zeros(len(train_idx)), labels[train_idx], groups[train_idx]
        ):
            global_train = train_idx[inner_train]
            global_test = train_idx[inner_test]
            assert_group_separation(global_train, global_test, groups)
            for k in candidates:
                probabilities = predict_knn_proba(
                    matrix, global_train, global_test, labels, k
                )
                predictions = predictions_from_probabilities(probabilities)
                scores[k].append(
                    balanced_accuracy_score(labels[global_test], predictions)
                )
    return max(candidates, key=lambda k: (np.mean(scores[k]), -k))


def evaluate_distance_metric(
    name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    bootstrap_iterations: int = 2000,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=outer_seed
    )
    predictions = np.empty(len(labels), dtype=labels.dtype)
    probabilities = np.zeros((len(labels), len(GENRES)), dtype=np.float64)
    fold_rows = []
    selected_k = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        assert_group_separation(train_idx, test_idx, groups)
        k = choose_k(
            matrix,
            train_idx,
            labels,
            groups,
            inner_seeds=inner_seeds,
        )
        fold_probabilities = predict_knn_proba(
            matrix, train_idx, test_idx, labels, k
        )
        fold_predictions = predictions_from_probabilities(fold_probabilities)
        predictions[test_idx] = fold_predictions
        probabilities[test_idx] = fold_probabilities
        selected_k.append(k)
        fold_rows.append(
            {
                "method": name,
                "outer_seed": outer_seed,
                "fold": fold,
                "k": k,
                "n_test": len(test_idx),
                **metric_values(labels[test_idx], fold_predictions),
            }
        )
    summary = summarize_predictions(
        name,
        labels,
        predictions,
        fold_rows,
        bootstrap_iterations,
        30_000 + outer_seed,
        {
            "k_mode": Counter(selected_k).most_common(1)[0][0],
            "k_selected": "/".join(map(str, selected_k)),
        },
    )
    return summary, predictions, probabilities, fold_rows


def make_random_forest(seed: int, n_estimators: int = 500) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def evaluate_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    compute_importance: bool = True,
    bootstrap_iterations: int = 2000,
    n_estimators: int = 500,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict], np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=outer_seed
    )
    predictions = np.empty(len(labels), dtype=labels.dtype)
    probabilities = np.zeros((len(labels), len(GENRES)), dtype=np.float64)
    fold_rows = []
    importances = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels, groups), start=1
    ):
        assert_group_separation(train_idx, test_idx, groups)
        model = make_random_forest(
            10_000 + outer_seed * 10 + fold, n_estimators=n_estimators
        )
        model.fit(features[train_idx], labels[train_idx])
        fold_probabilities = _align_probabilities(
            model.predict_proba(features[test_idx]), model.classes_
        )
        fold_predictions = predictions_from_probabilities(fold_probabilities)
        predictions[test_idx] = fold_predictions
        probabilities[test_idx] = fold_probabilities
        if compute_importance:
            importance = permutation_importance(
                model,
                features[test_idx],
                labels[test_idx],
                scoring="balanced_accuracy",
                n_repeats=10,
                random_state=20_000 + outer_seed * 10 + fold,
                n_jobs=-1,
            )
            importances.append(importance.importances_mean)
        fold_rows.append(
            {
                "method": "RF descriptors",
                "outer_seed": outer_seed,
                "fold": fold,
                "k": "",
                "n_test": len(test_idx),
                **metric_values(labels[test_idx], fold_predictions),
            }
        )
    summary = summarize_predictions(
        "RF descriptors",
        labels,
        predictions,
        fold_rows,
        bootstrap_iterations,
        40_000 + outer_seed,
        {"k_mode": "", "k_selected": ""},
    )
    mean_importance = (
        np.mean(importances, axis=0)
        if importances
        else np.full(features.shape[1], np.nan)
    )
    return summary, predictions, probabilities, fold_rows, mean_importance


def evaluate_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    bootstrap_iterations: int = 2000,
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=outer_seed
    )
    predictions = np.empty(len(labels), dtype=labels.dtype)
    probabilities = np.zeros((len(labels), len(GENRES)), dtype=np.float64)
    fold_rows = []
    selected_c = []
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(features, labels, groups), start=1
    ):
        assert_group_separation(train_idx, test_idx, groups)
        scores = {value: [] for value in c_values}
        for inner_seed in inner_seeds:
            inner = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=inner_seed
            )
            for inner_train, inner_test in inner.split(
                features[train_idx], labels[train_idx], groups[train_idx]
            ):
                global_train = train_idx[inner_train]
                global_test = train_idx[inner_test]
                assert_group_separation(global_train, global_test, groups)
                for c_value in c_values:
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=c_value,
                            max_iter=5000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=50_000 + inner_seed,
                        ),
                    )
                    model.fit(features[global_train], labels[global_train])
                    prediction = model.predict(features[global_test])
                    scores[c_value].append(
                        balanced_accuracy_score(labels[global_test], prediction)
                    )
        selected = max(c_values, key=lambda value: (np.mean(scores[value]), -value))
        selected_c.append(selected)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=selected,
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=60_000 + outer_seed * 10 + fold,
            ),
        )
        model.fit(features[train_idx], labels[train_idx])
        raw_probabilities = model.predict_proba(features[test_idx])
        fold_probabilities = _align_probabilities(
            raw_probabilities, model.named_steps["logisticregression"].classes_
        )
        fold_predictions = predictions_from_probabilities(fold_probabilities)
        predictions[test_idx] = fold_predictions
        probabilities[test_idx] = fold_probabilities
        fold_rows.append(
            {
                "method": "Multinomial logistic descriptors",
                "outer_seed": outer_seed,
                "fold": fold,
                "k": "",
                "selected_c": selected,
                "n_test": len(test_idx),
                **metric_values(labels[test_idx], fold_predictions),
            }
        )
    summary = summarize_predictions(
        "Multinomial logistic descriptors",
        labels,
        predictions,
        fold_rows,
        bootstrap_iterations,
        70_000 + outer_seed,
        {
            "c_selected": "/".join(f"{value:g}" for value in selected_c),
            "k_mode": "",
            "k_selected": "",
        },
    )
    return summary, predictions, probabilities, fold_rows


def evaluate_tuned_distance_grid(
    method: str,
    matrices: dict[tuple, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    parameter_names: tuple[str, ...],
    outer_seed: int = 42,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    bootstrap_iterations: int = 2000,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=outer_seed
    )
    predictions = np.empty(len(labels), dtype=labels.dtype)
    probabilities = np.zeros((len(labels), len(GENRES)), dtype=np.float64)
    fold_rows = []
    selections: list[tuple] = []
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(labels)), labels, groups), start=1
    ):
        assert_group_separation(train_idx, test_idx, groups)
        scores = {
            (*parameters, k): []
            for parameters in matrices
            for k in K_CANDIDATES
        }
        for inner_seed in inner_seeds:
            inner = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=inner_seed
            )
            for inner_train, inner_test in inner.split(
                np.zeros(len(train_idx)), labels[train_idx], groups[train_idx]
            ):
                global_train = train_idx[inner_train]
                global_test = train_idx[inner_test]
                assert_group_separation(global_train, global_test, groups)
                for parameters, matrix in matrices.items():
                    for k in K_CANDIDATES:
                        fold_probabilities = predict_knn_proba(
                            matrix, global_train, global_test, labels, k
                        )
                        fold_predictions = predictions_from_probabilities(
                            fold_probabilities
                        )
                        scores[(*parameters, k)].append(
                            balanced_accuracy_score(
                                labels[global_test], fold_predictions
                            )
                        )
        selected = max(
            scores,
            key=lambda item: (
                np.mean(scores[item]),
                *tuple(-float(value) for value in item),
            ),
        )
        parameters = selected[:-1]
        k = int(selected[-1])
        selections.append(selected)
        fold_probabilities = predict_knn_proba(
            matrices[parameters], train_idx, test_idx, labels, k
        )
        fold_predictions = predictions_from_probabilities(fold_probabilities)
        predictions[test_idx] = fold_predictions
        probabilities[test_idx] = fold_probabilities
        row = {
            "method": method,
            "outer_seed": outer_seed,
            "fold": fold,
            "k": k,
            "n_test": len(test_idx),
            **metric_values(labels[test_idx], fold_predictions),
        }
        row.update(dict(zip(parameter_names, parameters)))
        fold_rows.append(row)

    extra = {
        "k_mode": Counter(int(item[-1]) for item in selections).most_common(1)[0][0],
        "k_selected": "/".join(str(int(item[-1])) for item in selections),
    }
    for index, name in enumerate(parameter_names):
        extra[f"{name}_selected"] = "/".join(
            f"{item[index]:g}" for item in selections
        )
    summary = summarize_predictions(
        method,
        labels,
        predictions,
        fold_rows,
        bootstrap_iterations,
        80_000 + outer_seed,
        extra,
    )
    return summary, predictions, probabilities, fold_rows


def evaluate_tuned_mhd(
    matrices: dict[tuple[int, float], np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    bootstrap_iterations: int = 2000,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    return evaluate_tuned_distance_grid(
        "Tuned MHD (nested)",
        matrices,
        labels,
        groups,
        ("resample_points", "velocity_weight"),
        outer_seed,
        inner_seeds,
        bootstrap_iterations,
    )


def evaluate_tuned_dtw(
    matrices: dict[tuple[int, float, float], np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    bootstrap_iterations: int = 2000,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    return evaluate_tuned_distance_grid(
        "Multivariate DTW (nested)",
        matrices,
        labels,
        groups,
        ("resample_points", "velocity_weight", "window_fraction"),
        outer_seed,
        inner_seeds,
        bootstrap_iterations,
    )


def evaluate_fusion(
    mhd_matrices: dict[tuple[int, float], np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    alphas: tuple[float, ...] = (0.25, 0.50, 0.75),
    bootstrap_iterations: int = 2000,
    n_estimators: int = 500,
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    """Jointly select MHD parameters, K, and RF probability weight."""
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=outer_seed
    )
    predictions = np.empty(len(labels), dtype=labels.dtype)
    probabilities = np.zeros((len(labels), len(GENRES)), dtype=np.float64)
    fold_rows = []
    selections = []
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(features, labels, groups), start=1
    ):
        assert_group_separation(train_idx, test_idx, groups)
        scores = {
            (*parameters, k, alpha): []
            for parameters in mhd_matrices
            for k in K_CANDIDATES
            for alpha in alphas
        }
        inner_counter = 0
        for inner_seed in inner_seeds:
            inner = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=inner_seed
            )
            for inner_train, inner_test in inner.split(
                features[train_idx], labels[train_idx], groups[train_idx]
            ):
                global_train = train_idx[inner_train]
                global_test = train_idx[inner_test]
                assert_group_separation(global_train, global_test, groups)
                rf = make_random_forest(
                    90_000 + outer_seed * 100 + fold * 10 + inner_counter,
                    n_estimators=max(250, n_estimators // 2),
                )
                rf.fit(features[global_train], labels[global_train])
                rf_probabilities = _align_probabilities(
                    rf.predict_proba(features[global_test]), rf.classes_
                )
                for parameters, matrix in mhd_matrices.items():
                    for k in K_CANDIDATES:
                        mhd_probabilities = predict_knn_proba(
                            matrix, global_train, global_test, labels, k
                        )
                        for alpha in alphas:
                            fused = (
                                alpha * rf_probabilities
                                + (1.0 - alpha) * mhd_probabilities
                            )
                            fused_predictions = predictions_from_probabilities(fused)
                            scores[(*parameters, k, alpha)].append(
                                balanced_accuracy_score(
                                    labels[global_test], fused_predictions
                                )
                            )
                inner_counter += 1
        selected = max(
            scores,
            key=lambda item: (
                np.mean(scores[item]),
                -float(item[0]),
                -float(item[1]),
                -float(item[2]),
                -float(item[3]),
            ),
        )
        n_points, weight, k, alpha = selected
        selections.append(selected)
        rf = make_random_forest(
            100_000 + outer_seed * 10 + fold, n_estimators=n_estimators
        )
        rf.fit(features[train_idx], labels[train_idx])
        rf_probabilities = _align_probabilities(
            rf.predict_proba(features[test_idx]), rf.classes_
        )
        mhd_probabilities = predict_knn_proba(
            mhd_matrices[(int(n_points), float(weight))],
            train_idx,
            test_idx,
            labels,
            int(k),
        )
        fold_probabilities = (
            float(alpha) * rf_probabilities
            + (1.0 - float(alpha)) * mhd_probabilities
        )
        fold_predictions = predictions_from_probabilities(fold_probabilities)
        predictions[test_idx] = fold_predictions
        probabilities[test_idx] = fold_probabilities
        fold_rows.append(
            {
                "method": "MHD-RF probability fusion",
                "outer_seed": outer_seed,
                "fold": fold,
                "k": int(k),
                "n_test": len(test_idx),
                "resample_points": int(n_points),
                "velocity_weight": float(weight),
                "rf_weight": float(alpha),
                **metric_values(labels[test_idx], fold_predictions),
            }
        )
    summary = summarize_predictions(
        "MHD-RF probability fusion",
        labels,
        predictions,
        fold_rows,
        bootstrap_iterations,
        110_000 + outer_seed,
        {
            "k_mode": Counter(int(item[2]) for item in selections).most_common(1)[0][0],
            "k_selected": "/".join(str(int(item[2])) for item in selections),
            "resample_points_selected": "/".join(
                str(int(item[0])) for item in selections
            ),
            "velocity_weight_selected": "/".join(
                f"{float(item[1]):.2f}" for item in selections
            ),
            "rf_weight_selected": "/".join(
                f"{float(item[3]):.2f}" for item in selections
            ),
        },
    )
    return summary, predictions, probabilities, fold_rows


def distance_statistics(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    permutations: int = 9999,
) -> dict:
    upper = np.triu_indices(len(labels), k=1)
    distances = matrix[upper]
    same = labels[upper[0]] == labels[upper[1]]
    within = distances[same]
    between = distances[~same]
    observed_gap = float(between.mean() - within.mean())
    auc = float(roc_auc_score(same.astype(int), -distances))

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_labels = np.empty(len(unique_groups), dtype=labels.dtype)
    for group_index in range(len(unique_groups)):
        values = np.unique(labels[inverse == group_index])
        if len(values) != 1:
            raise ValueError(f"group {unique_groups[group_index]} spans labels")
        group_labels[group_index] = values[0]
    rng = np.random.default_rng(seed)
    permuted_gaps = np.empty(permutations)
    for iteration in range(permutations):
        shuffled = rng.permutation(group_labels)[inverse]
        perm_same = shuffled[upper[0]] == shuffled[upper[1]]
        permuted_gaps[iteration] = (
            distances[~perm_same].mean() - distances[perm_same].mean()
        )
    p_value = (1 + np.sum(permuted_gaps >= observed_gap)) / (permutations + 1)
    return {
        "within_mean": float(within.mean()),
        "within_median": float(np.median(within)),
        "between_mean": float(between.mean()),
        "between_median": float(np.median(between)),
        "mean_gap": observed_gap,
        "pair_auc": auc,
        "permutation_p": float(p_value),
        "permutation_count": permutations,
        "permutation_groups": len(unique_groups),
        "within": within,
        "between": between,
    }


def mcnemar_holm(
    labels: np.ndarray, predictions: dict[str, np.ndarray]
) -> list[dict]:
    names = list(predictions)
    rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_correct = predictions[left] == labels
            right_correct = predictions[right] == labels
            left_only = int(np.sum(left_correct & ~right_correct))
            right_only = int(np.sum(~left_correct & right_correct))
            discordant = left_only + right_only
            p_value = (
                float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "a_correct_b_wrong": left_only,
                    "a_wrong_b_correct": right_only,
                    "discordant": discordant,
                    "mcnemar_exact_p": p_value,
                }
            )
    order = np.argsort([row["mcnemar_exact_p"] for row in rows])
    adjusted = np.ones(len(rows))
    running = 0.0
    for rank, row_index in enumerate(order):
        value = min(1.0, (len(rows) - rank) * rows[row_index]["mcnemar_exact_p"])
        running = max(running, value)
        adjusted[row_index] = running
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(value)
        row["significant_0_05"] = int(value < 0.05)
    return rows


def feature_ablation(
    features: np.ndarray,
    feature_names: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seed: int = 42,
    n_estimators: int = 400,
) -> list[dict]:
    groups_by_name = feature_group_indices(feature_names)
    all_indices = np.arange(features.shape[1])
    rows = []
    full_summary, _, _, _, _ = evaluate_random_forest(
        features,
        labels,
        groups,
        outer_seed=outer_seed,
        compute_importance=False,
        bootstrap_iterations=0,
        n_estimators=n_estimators,
    )
    rows.append(
        {
            "feature_set": "all",
            "removed_group": "none",
            "n_features": features.shape[1],
            **{key: full_summary[key] for key in ("accuracy", "balanced_accuracy", "macro_f1")},
        }
    )
    for name, removed in groups_by_name.items():
        retained = np.setdiff1d(all_indices, removed)
        summary, _, _, _, _ = evaluate_random_forest(
            features[:, retained],
            labels,
            groups,
            outer_seed=outer_seed,
            compute_importance=False,
            bootstrap_iterations=0,
            n_estimators=n_estimators,
        )
        rows.append(
            {
                "feature_set": f"all_except_{name}",
                "removed_group": name,
                "n_features": len(retained),
                **{
                    key: summary[key]
                    for key in ("accuracy", "balanced_accuracy", "macro_f1")
                },
            }
        )
    baseline = rows[0]["balanced_accuracy"]
    for row in rows:
        row["balanced_accuracy_change"] = row["balanced_accuracy"] - baseline
    return rows


def class_recall_rows(
    labels: np.ndarray, predictions: dict[str, np.ndarray]
) -> list[dict]:
    rows = []
    for method, values in predictions.items():
        matrix = confusion_matrix(labels, values, labels=GENRES, normalize="true")
        for index, genre in enumerate(GENRES):
            rows.append(
                {"method": method, "genre": genre, "recall": float(matrix[index, index])}
            )
    return rows


def disagreement_rows(
    filenames: np.ndarray,
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for index in range(len(labels)):
        unique_predictions = {str(values[index]) for values in predictions.values()}
        any_error = any(values[index] != labels[index] for values in predictions.values())
        if len(unique_predictions) == 1 and not any_error:
            continue
        row = {
            "file": str(filenames[index]),
            "true_genre": str(labels[index]),
            "model_disagreement": int(len(unique_predictions) > 1),
        }
        for method, values in predictions.items():
            row[f"{method}_prediction"] = str(values[index])
            row[f"{method}_confidence"] = float(np.max(probabilities[method][index]))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["model_disagreement"],
            row["true_genre"],
            row["file"].casefold(),
        )
    )
    return rows


def repeated_validation(
    mhd_matrices: dict[tuple[int, float], np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_seeds: tuple[int, ...] = REPEAT_OUTER_SEEDS,
    inner_seeds: tuple[int, ...] = INNER_SEEDS,
    n_estimators: int = 400,
) -> tuple[list[dict], list[dict]]:
    rows = []
    for outer_seed in outer_seeds:
        tuned, _, _, _ = evaluate_tuned_mhd(
            mhd_matrices,
            labels,
            groups,
            outer_seed=outer_seed,
            inner_seeds=inner_seeds,
            bootstrap_iterations=0,
        )
        rf, _, _, _, _ = evaluate_random_forest(
            features,
            labels,
            groups,
            outer_seed=outer_seed,
            compute_importance=False,
            bootstrap_iterations=0,
            n_estimators=n_estimators,
        )
        fusion, _, _, _ = evaluate_fusion(
            mhd_matrices,
            features,
            labels,
            groups,
            outer_seed=outer_seed,
            inner_seeds=inner_seeds,
            bootstrap_iterations=0,
            n_estimators=n_estimators,
        )
        for summary in (tuned, rf, fusion):
            rows.append(
                {
                    "method": summary["method"],
                    "outer_seed": outer_seed,
                    **{
                        key: summary[key]
                        for key in (
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                            "fold_mean",
                            "fold_std",
                        )
                    },
                }
            )

    aggregate = []
    for method in (
        "Tuned MHD (nested)",
        "RF descriptors",
        "MHD-RF probability fusion",
    ):
        selected = [row for row in rows if row["method"] == method]
        accuracy = np.asarray([row["accuracy"] for row in selected])
        macro_f1 = np.asarray([row["macro_f1"] for row in selected])
        aggregate.append(
            {
                "method": method,
                "repeats": len(selected),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                "accuracy_min": float(accuracy.min()),
                "accuracy_max": float(accuracy.max()),
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_std": (
                    float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0
                ),
            }
        )
    return rows, aggregate
