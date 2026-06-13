"""Run the version-3 leakage-safe melody geometry experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    GENRES,
    PIPELINE_VERSION,
    RESULTS_DIR,
    ROOT,
    TABLE_DIR,
    dataset_cache_path,
    ensure_dirs,
    read_csv,
    sample_dataset,
    save_csv,
)
from distance_models import (  # noqa: E402
    build_dtw_grid,
    build_mhd_grid,
    compute_base_distances,
    distance_cache_path,
    sensitivity_cache_path,
)
from evaluation import (  # noqa: E402
    INNER_SEEDS,
    REPEAT_OUTER_SEEDS,
    class_recall_rows,
    disagreement_rows,
    distance_statistics,
    evaluate_distance_metric,
    evaluate_fusion,
    evaluate_logistic_regression,
    evaluate_random_forest,
    evaluate_tuned_dtw,
    evaluate_tuned_mhd,
    feature_ablation,
    mcnemar_holm,
    repeated_validation,
)
from midi_geometry import nearest_distance_summaries, relative_curve  # noqa: E402
from provenance import (  # noqa: E402
    artifact_inventory,
    combined_digest,
    environment_snapshot,
    write_manifest,
)


PREDICTION_FILE = RESULTS_DIR / "predictions_primary.npz"
SUMMARY_FILE = RESULTS_DIR / "summary.json"
MANIFEST_FILE = RESULTS_DIR / "run_manifest.json"
CONFIG_FILE = RESULTS_DIR / "experiment_config.json"
LATEX_VALUES_FILE = ROOT / "report_clk" / "generated_results.tex"


def genre_mean_matrix(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(GENRES), len(GENRES)), dtype=np.float64)
    for row, genre_a in enumerate(GENRES):
        index_a = np.flatnonzero(labels == genre_a)
        for column, genre_b in enumerate(GENRES):
            index_b = np.flatnonzero(labels == genre_b)
            block = matrix[np.ix_(index_a, index_b)]
            if row == column:
                block = block[np.triu_indices_from(block, k=1)]
            result[row, column] = block.mean()
    return result


def synthetic_robustness() -> list[dict]:
    phase = np.linspace(0, 1, 96)
    base = np.column_stack(
        (phase, 0.55 * np.sin(4 * np.pi * phase), np.zeros_like(phase))
    )
    rows = []
    for amplitude in np.linspace(0, 2.0, 11):
        perturbed = base.copy()
        perturbed[48, 1] += amplitude
        hd, q95, mhd = nearest_distance_summaries(base, perturbed)
        rows.append(
            {
                "outlier_amplitude": float(amplitude),
                "hd": hd,
                "q95": q95,
                "mhd": mhd,
            }
        )
    return rows


def sensitivity_rows(
    matrices: dict[tuple[int, float], np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    base_points: int,
    inner_seeds: tuple[int, ...],
) -> list[dict]:
    rows = []
    point_values = sorted({points for points, _ in matrices})
    weight_values = sorted({weight for _, weight in matrices})
    for points in point_values:
        key = (points, 0.25)
        if key not in matrices:
            continue
        summary, _, _, _ = evaluate_distance_metric(
            f"MHD points={points}",
            matrices[key],
            labels,
            groups,
            inner_seeds=inner_seeds,
            bootstrap_iterations=0,
        )
        rows.append(
            {
                "parameter": "resample_points",
                "value": points,
                "fold_mean": summary["fold_mean"],
                "fold_std": summary["fold_std"],
            }
        )
    for weight in weight_values:
        key = (base_points, weight)
        if key not in matrices:
            continue
        summary, _, _, _ = evaluate_distance_metric(
            f"MHD velocity={weight:.2f}",
            matrices[key],
            labels,
            groups,
            inner_seeds=inner_seeds,
            bootstrap_iterations=0,
        )
        rows.append(
            {
                "parameter": "velocity_weight",
                "value": weight,
                "fold_mean": summary["fold_mean"],
                "fold_std": summary["fold_std"],
            }
        )
    return rows


def latex_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def model_by_name(summaries: list[dict], method: str) -> dict:
    return next(row for row in summaries if row["method"] == method)


def write_latex_values(
    dataset: dict[str, np.ndarray],
    summaries: list[dict],
    statistics: dict,
    repeated_summary: list[dict],
    significance: list[dict],
) -> None:
    actual = int(dataset["per_genre_actual"])
    tuned = model_by_name(summaries, "Tuned MHD (nested)")
    dtw = model_by_name(summaries, "Multivariate DTW (nested)")
    logistic = model_by_name(summaries, "Multinomial logistic descriptors")
    rf = model_by_name(summaries, "RF descriptors")
    fusion = model_by_name(summaries, "MHD-RF probability fusion")
    phase = model_by_name(summaries, "Phase-aligned trajectory RMSE")
    repeated = {row["method"]: row for row in repeated_summary}
    audit_total = next(
        row
        for row in read_csv(TABLE_DIR / "data_audit_summary.csv")
        if row["genre"] == "TOTAL"
    )
    model_commands = {
        "HDLocal": model_by_name(summaries, "HD local min-max"),
        "HDTP": model_by_name(summaries, "HD relative TP"),
        "HDTPV": model_by_name(summaries, "HD relative TPV"),
        "QHD": model_by_name(summaries, "Q95-HD relative TPV"),
        "FixedMHD": model_by_name(summaries, "MHD relative TPV"),
        "PhaseRMSE": phase,
        "TunedMHD": tuned,
        "DTW": dtw,
        "Logistic": logistic,
        "RF": rf,
        "Fusion": fusion,
    }

    commands = {
        "PipelineVersion": PIPELINE_VERSION,
        "SampleCount": str(len(dataset["labels"])),
        "PerGenreCount": str(actual),
        "CrossGenreGroupCount": str(int(dataset["cross_genre_group_count"])),
        "RawFileCount": audit_total["raw_files"],
        "FingerprintFailureCount": audit_total["fingerprint_failures"],
        "TitleDuplicateSetCount": audit_total["title_duplicate_sets"],
        "MelodyDuplicateSetCount": audit_total["melody_duplicate_sets"],
        "UnionGroupCount": audit_total["union_groups"],
        "CrossGenreFileCount": audit_total["cross_genre_files"],
        "WithinMHD": latex_number(statistics["within_mean"], 4),
        "BetweenMHD": latex_number(statistics["between_mean"], 4),
        "MHDGap": latex_number(statistics["mean_gap"], 4),
        "PairAUC": latex_number(statistics["pair_auc"], 4),
        "PermutationP": latex_number(statistics["permutation_p"], 4),
        "PermutationCount": str(statistics["permutation_count"]),
        "TunedMHDAcc": latex_number(100 * tuned["accuracy"]),
        "TunedMHDLow": latex_number(100 * tuned["accuracy_ci_low"]),
        "TunedMHDHigh": latex_number(100 * tuned["accuracy_ci_high"]),
        "DTWAcc": latex_number(100 * dtw["accuracy"]),
        "PhaseRMSEAcc": latex_number(100 * phase["accuracy"]),
        "LogisticAcc": latex_number(100 * logistic["accuracy"]),
        "RFAcc": latex_number(100 * rf["accuracy"]),
        "RFLow": latex_number(100 * rf["accuracy_ci_low"]),
        "RFHigh": latex_number(100 * rf["accuracy_ci_high"]),
        "FusionAcc": latex_number(100 * fusion["accuracy"]),
        "FusionLow": latex_number(100 * fusion["accuracy_ci_low"]),
        "FusionHigh": latex_number(100 * fusion["accuracy_ci_high"]),
        "RepeatedMHDAcc": latex_number(
            100 * repeated["Tuned MHD (nested)"]["accuracy_mean"]
        ),
        "RepeatedRFAcc": latex_number(
            100 * repeated["RF descriptors"]["accuracy_mean"]
        ),
        "RepeatedFusionAcc": latex_number(
            100 * repeated["MHD-RF probability fusion"]["accuracy_mean"]
        ),
        "RepeatedMHDStd": latex_number(
            100 * repeated["Tuned MHD (nested)"]["accuracy_std"]
        ),
        "RepeatedRFStd": latex_number(
            100 * repeated["RF descriptors"]["accuracy_std"]
        ),
        "RepeatedFusionStd": latex_number(
            100 * repeated["MHD-RF probability fusion"]["accuracy_std"]
        ),
    }
    for prefix, row in model_commands.items():
        commands[f"{prefix}Acc"] = latex_number(100 * row["accuracy"])
        commands[f"{prefix}Low"] = latex_number(100 * row["accuracy_ci_low"])
        commands[f"{prefix}High"] = latex_number(100 * row["accuracy_ci_high"])
        commands[f"{prefix}FOne"] = latex_number(row["macro_f1"], 3)

    comparison_rows = {
        (row["model_a"], row["model_b"]): row for row in significance
    }
    if comparison_rows:
        mhd_rf = comparison_rows[("Tuned MHD (nested)", "RF descriptors")]
        rf_fusion = comparison_rows[
            ("RF descriptors", "MHD-RF probability fusion")
        ]
        commands["MHDvsRFHolmP"] = (
            "<0.0001"
            if mhd_rf["holm_adjusted_p"] < 0.0001
            else latex_number(mhd_rf["holm_adjusted_p"], 4)
        )
        commands["RFvsFusionHolmP"] = latex_number(
            rf_fusion["holm_adjusted_p"], 4
        )
    recall_prefixes = {
        "Tuned MHD (nested)": "TunedMHD",
        "RF descriptors": "RF",
        "MHD-RF probability fusion": "Fusion",
    }
    for row in read_csv(TABLE_DIR / "class_recall.csv"):
        prefix = recall_prefixes.get(row["method"])
        if prefix is not None:
            commands[f"{prefix}{row['genre']}Recall"] = latex_number(
                100 * float(row["recall"])
            )
    lines = [
        "% Auto-generated by code/final_experiment.py. Do not edit manually."
    ]
    for name, value in commands.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    LATEX_VALUES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_ready_statistics(statistics: dict) -> dict:
    return {
        key: value
        for key, value in statistics.items()
        if key not in {"within", "between"}
    }


def run_signature(summary: dict, predictions: dict[str, np.ndarray]) -> str:
    values = [json.dumps(summary["models"], sort_keys=True, ensure_ascii=True)]
    for name in sorted(predictions):
        values.extend(str(value) for value in predictions[name])
    return combined_digest(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe melody geometry experiment (pipeline v3)"
    )
    parser.add_argument("--per-genre", type=int, default=100)
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, min(16, os.cpu_count() or 2)),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--rebuild-distances", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    force = args.force_recompute or args.rebuild_data or args.rebuild_distances

    if args.quick:
        requested_per_genre = min(args.per_genre, 12)
        base_points = min(args.points, 48)
        inner_seeds = (2026,)
        repeat_seeds = (42,)
        bootstrap_iterations = 300
        mhd_points = tuple(sorted({24, 36, base_points}))
        mhd_weights = (0.0, 0.25, 0.50)
        dtw_points = (24, 36)
        dtw_weights = (0.0, 0.50)
        dtw_windows = (0.15,)
        n_estimators = 120
    else:
        requested_per_genre = args.per_genre
        base_points = args.points
        inner_seeds = INNER_SEEDS
        repeat_seeds = REPEAT_OUTER_SEEDS
        bootstrap_iterations = 2000
        mhd_points = (48, 72, 96)
        mhd_weights = (0.0, 0.10, 0.25, 0.50)
        dtw_points = (36, 48)
        dtw_weights = (0.0, 0.25, 0.50)
        dtw_windows = (0.10, 0.20)
        n_estimators = 500

    ensure_dirs()
    started = time.time()
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "genres": list(GENRES),
        "requested_per_genre": requested_per_genre,
        "base_points": base_points,
        "seed": args.seed,
        "workers": args.workers,
        "quick": args.quick,
        "force_recompute": force,
        "inner_seeds": list(inner_seeds),
        "outer_seeds": list(repeat_seeds),
        "bootstrap_iterations": bootstrap_iterations,
        "mhd_grid": {
            "points": list(mhd_points),
            "velocity_weights": list(mhd_weights),
            "k": [1, 3, 5, 7, 9],
        },
        "dtw_grid": {
            "points": list(dtw_points),
            "velocity_weights": list(dtw_weights),
            "window_fractions": list(dtw_windows),
            "k": [1, 3, 5, 7, 9],
        },
        "fusion_rf_weights": [0.25, 0.50, 0.75],
        "primary_outer_seed": 42,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    initial_environment = environment_snapshot(ROOT)
    print(
        f"[config] v3, target={requested_per_genre}/genre, "
        f"points={base_points}, quick={args.quick}"
    )

    dataset = sample_dataset(
        requested_per_genre,
        base_points,
        args.seed,
        args.workers,
        force=force,
    )
    curves = dataset["curves"]
    labels = dataset["labels"]
    groups = dataset["groups"]
    features = dataset["features"]
    filenames = dataset["filenames"]
    actual_per_genre = int(dataset["per_genre_actual"])

    base_matrices = compute_base_distances(
        curves,
        actual_per_genre,
        base_points,
        args.seed,
        args.workers,
        force=force,
    )
    mhd_matrices = build_mhd_grid(
        curves,
        labels,
        filenames,
        base_matrices["mhd_tpv"],
        actual_per_genre,
        args.seed,
        args.workers,
        point_counts=mhd_points,
        velocity_weights=mhd_weights,
        force=force,
    )
    dtw_matrices = build_dtw_grid(
        labels,
        filenames,
        actual_per_genre,
        args.seed,
        args.workers,
        point_counts=dtw_points,
        velocity_weights=dtw_weights,
        window_fractions=dtw_windows,
        force=force,
    )

    summaries: list[dict] = []
    fold_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    baseline_methods = {
        "hd_local": "HD local min-max",
        "hd_tp": "HD relative TP",
        "hd_tpv": "HD relative TPV",
        "q95_tpv": "Q95-HD relative TPV",
        "mhd_tpv": "MHD relative TPV",
        "phase_rmse": "Phase-aligned trajectory RMSE",
    }
    for key, method in baseline_methods.items():
        summary, prediction, probability, folds = evaluate_distance_metric(
            method,
            base_matrices[key],
            labels,
            groups,
            inner_seeds=inner_seeds,
            bootstrap_iterations=bootstrap_iterations,
        )
        summaries.append(summary)
        fold_rows.extend(folds)
        predictions[method] = prediction
        probabilities[method] = probability
        print(f"[eval] {method}: {summary['accuracy']:.4f}")

    tuned_mhd, tuned_prediction, tuned_probability, tuned_folds = evaluate_tuned_mhd(
        mhd_matrices,
        labels,
        groups,
        inner_seeds=inner_seeds,
        bootstrap_iterations=bootstrap_iterations,
    )
    summaries.append(tuned_mhd)
    fold_rows.extend(tuned_folds)
    predictions[tuned_mhd["method"]] = tuned_prediction
    probabilities[tuned_mhd["method"]] = tuned_probability

    tuned_dtw, dtw_prediction, dtw_probability, dtw_folds = evaluate_tuned_dtw(
        dtw_matrices,
        labels,
        groups,
        inner_seeds=inner_seeds,
        bootstrap_iterations=bootstrap_iterations,
    )
    summaries.append(tuned_dtw)
    fold_rows.extend(dtw_folds)
    predictions[tuned_dtw["method"]] = dtw_prediction
    probabilities[tuned_dtw["method"]] = dtw_probability

    logistic, logistic_prediction, logistic_probability, logistic_folds = (
        evaluate_logistic_regression(
            features,
            labels,
            groups,
            inner_seeds=inner_seeds,
            bootstrap_iterations=bootstrap_iterations,
        )
    )
    summaries.append(logistic)
    fold_rows.extend(logistic_folds)
    predictions[logistic["method"]] = logistic_prediction
    probabilities[logistic["method"]] = logistic_probability

    rf, rf_prediction, rf_probability, rf_folds, importances = (
        evaluate_random_forest(
            features,
            labels,
            groups,
            bootstrap_iterations=bootstrap_iterations,
            n_estimators=n_estimators,
        )
    )
    summaries.append(rf)
    fold_rows.extend(rf_folds)
    predictions[rf["method"]] = rf_prediction
    probabilities[rf["method"]] = rf_probability

    fusion, fusion_prediction, fusion_probability, fusion_folds = evaluate_fusion(
        mhd_matrices,
        features,
        labels,
        groups,
        inner_seeds=inner_seeds,
        bootstrap_iterations=bootstrap_iterations,
        n_estimators=n_estimators,
    )
    summaries.append(fusion)
    fold_rows.extend(fusion_folds)
    predictions[fusion["method"]] = fusion_prediction
    probabilities[fusion["method"]] = fusion_probability

    statistics = distance_statistics(
        base_matrices["mhd_tpv"],
        labels,
        groups,
        args.seed,
        permutations=999 if args.quick else 9999,
    )
    sensitivity = sensitivity_rows(
        mhd_matrices, labels, groups, base_points, inner_seeds
    )
    ablation = feature_ablation(
        features,
        dataset["feature_names"],
        labels,
        groups,
        n_estimators=max(120, n_estimators - 100),
    )
    primary_predictions = {
        method: predictions[method]
        for method in (
            "Tuned MHD (nested)",
            "RF descriptors",
            "MHD-RF probability fusion",
        )
    }
    significance = mcnemar_holm(labels, primary_predictions)
    repeated_rows, repeated_summary = repeated_validation(
        mhd_matrices,
        features,
        labels,
        groups,
        outer_seeds=repeat_seeds,
        inner_seeds=inner_seeds,
        n_estimators=max(120, n_estimators - 100),
    )
    genre_matrix = genre_mean_matrix(base_matrices["mhd_tpv"], labels)
    synthetic = synthetic_robustness()
    recalls = class_recall_rows(labels, primary_predictions)
    diagnostics = disagreement_rows(
        filenames,
        labels,
        primary_predictions,
        {method: probabilities[method] for method in primary_predictions},
    )

    save_csv(TABLE_DIR / "model_summary.csv", summaries)
    save_csv(TABLE_DIR / "fold_results.csv", fold_rows)
    save_csv(TABLE_DIR / "sensitivity.csv", sensitivity)
    save_csv(TABLE_DIR / "feature_ablation.csv", ablation)
    save_csv(TABLE_DIR / "mcnemar_holm.csv", significance)
    save_csv(TABLE_DIR / "class_recall.csv", recalls)
    save_csv(TABLE_DIR / "model_disagreements.csv", diagnostics)
    save_csv(TABLE_DIR / "repeated_validation.csv", repeated_rows)
    save_csv(TABLE_DIR / "repeated_validation_summary.csv", repeated_summary)
    save_csv(TABLE_DIR / "synthetic_robustness.csv", synthetic)
    save_csv(TABLE_DIR / "distance_statistics.csv", [json_ready_statistics(statistics)])
    np.savetxt(
        TABLE_DIR / "genre_mean_distance.csv",
        genre_matrix,
        delimiter=",",
        header=",".join(GENRES),
        comments="",
    )

    prediction_payload = {
        "labels": labels,
        "filenames": filenames,
        "feature_names": dataset["feature_names"],
        "rf_importances": importances,
    }
    for method, values in predictions.items():
        key = (
            method.lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        prediction_payload[f"{key}_predictions"] = values
        prediction_payload[f"{key}_probabilities"] = probabilities[method]
    np.savez_compressed(PREDICTION_FILE, **prediction_payload)

    summary = {
        "config": config,
        "pipeline_version": PIPELINE_VERSION,
        "genres": list(GENRES),
        "n_samples": len(labels),
        "per_genre_actual": actual_per_genre,
        "class_counts": {
            genre: int(np.sum(labels == genre)) for genre in GENRES
        },
        "unique_groups": len(np.unique(groups)),
        "excluded_cross_genre_groups": int(dataset["cross_genre_group_count"]),
        "primary_geometry_model": "Tuned MHD (nested)",
        "distance_statistics_mhd": json_ready_statistics(statistics),
        "models": summaries,
        "sensitivity": sensitivity,
        "feature_ablation": ablation,
        "mcnemar_holm": significance,
        "repeated_validation": repeated_summary,
        "runtime_seconds": time.time() - started,
    }
    signature = run_signature(summary, primary_predictions)
    signature_path = RESULTS_DIR / "determinism_signature.json"
    previous_signature = None
    if signature_path.exists():
        previous_signature = json.loads(
            signature_path.read_text(encoding="utf-8")
        ).get("signature")
    determinism = {
        "signature": signature,
        "previous_signature": previous_signature,
        "matches_previous_run": (
            None if previous_signature is None else signature == previous_signature
        ),
    }
    signature_path.write_text(
        json.dumps(determinism, indent=2), encoding="utf-8"
    )
    summary["determinism"] = determinism
    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_latex_values(
        dataset, summaries, statistics, repeated_summary, significance
    )

    manifest = {
        "purpose": (
            "Final course-paper evaluation of leakage-safe melody geometry, "
            "descriptor, and fusion models."
        ),
        "decision_owner": "project author",
        "model_selection_rationale": (
            "Tuned MHD is the prespecified primary geometry model; other distance "
            "families are controls. RF and probability fusion test complementary "
            "descriptor information without replacing the Hausdorff research question."
        ),
        "command": " ".join(sys.argv),
        "config": config,
        "environment_at_start": initial_environment,
        "environment_at_end": environment_snapshot(ROOT),
        "data": {
            "dataset_root": str(DATA_DIR),
            "selected_file_digest": combined_digest(dataset["file_sha256"].tolist()),
            "selected_melody_digest": combined_digest(
                dataset["melody_fingerprints"].tolist()
            ),
            "selected_samples": len(labels),
            "unique_groups": len(np.unique(groups)),
        },
        "metrics_file": str(SUMMARY_FILE.relative_to(ROOT)),
        "runtime_seconds": time.time() - started,
        "artifacts": artifact_inventory(
            [
                SUMMARY_FILE,
                CONFIG_FILE,
                PREDICTION_FILE,
                LATEX_VALUES_FILE,
                TABLE_DIR,
            ],
            ROOT,
        ),
        "known_limits": [
            "single ADL Piano MIDI corpus",
            "skyline melody is a heuristic voice extractor",
            "genre labels are broad and potentially noisy",
            "the study supports predictive association, not causal interpretation",
        ],
    }
    write_manifest(MANIFEST_FILE, manifest)

    print(
        f"[done] samples={len(labels)}, tuned MHD={tuned_mhd['accuracy']:.4f}, "
        f"RF={rf['accuracy']:.4f}, fusion={fusion['accuracy']:.4f}"
    )
    print(f"[done] results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
