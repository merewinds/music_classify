"""Focused regression tests for the final melody-geometry pipeline."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .data_pipeline import (
        build_duplicate_groups,
        melody_fingerprint,
        sampled_group_order,
    )
    from .distance_models import _dtw_python, validate_distance_matrix
    from .evaluation import (
        assert_group_separation,
        aggregate_validation_runs,
        evaluate_distance_metric,
        paired_bootstrap_model_differences,
        predict_knn_proba,
        predict_knn_proba_candidates,
    )
    from .midi_geometry import (
        _read_vlq,
        canonical_title,
        feature_group_indices,
        melody_features,
        nearest_distance_summaries,
        parse_midi_note_ons,
        skyline_melody,
    )
    from .paper_assets import preserve_method_taxonomy, sha256_file
except ImportError:
    from data_pipeline import (
        build_duplicate_groups,
        melody_fingerprint,
        sampled_group_order,
    )
    from distance_models import _dtw_python, validate_distance_matrix
    from evaluation import (
        assert_group_separation,
        aggregate_validation_runs,
        evaluate_distance_metric,
        paired_bootstrap_model_differences,
        predict_knn_proba,
        predict_knn_proba_candidates,
    )
    from midi_geometry import (
        _read_vlq,
        canonical_title,
        feature_group_indices,
        melody_features,
        nearest_distance_summaries,
        parse_midi_note_ons,
        skyline_melody,
    )
    from paper_assets import preserve_method_taxonomy, sha256_file


def midi_file(midi_format: int, track: bytes, division: int = 480) -> bytes:
    header = struct.pack(">4sIHHH", b"MThd", 6, midi_format, 1, division)
    return header + struct.pack(">4sI", b"MTrk", len(track)) + track


class PipelineTests(unittest.TestCase):
    def test_vlq_rejects_more_than_four_bytes(self) -> None:
        with self.assertRaises(ValueError):
            _read_vlq(b"\x81\x80\x80\x80\x00", 0)

    def test_format_two_is_rejected(self) -> None:
        track = b"\x00\xff\x2f\x00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "format2.mid"
            path.write_bytes(midi_file(2, track))
            with self.assertRaisesRegex(ValueError, "format 2"):
                parse_midi_note_ons(path)

    def test_running_status_note_ons_are_parsed(self) -> None:
        track = (
            b"\x00\x90\x3c\x40"
            b"\x00\x40\x50"
            b"\x00\xff\x2f\x00"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "running.mid"
            path.write_bytes(midi_file(0, track))
            notes, division = parse_midi_note_ons(path)
        self.assertEqual(division, 480)
        np.testing.assert_array_equal(
            notes,
            np.asarray([[0, 60, 64], [0, 64, 80]], dtype=float),
        )

    def test_skyline_breaks_pitch_ties_by_velocity(self) -> None:
        note_ons = np.asarray(
            [
                [0, 72, 32],
                [0, 72, 96],
                [0, 67, 127],
                [480, 70, 55],
            ],
            dtype=float,
        )
        melody = skyline_melody(note_ons, division=480)
        np.testing.assert_array_equal(
            melody,
            np.asarray([[0, 72, 96], [1, 70, 55]], dtype=float),
        )

    def test_title_keeps_work_number_but_removes_version_suffix(self) -> None:
        self.assertEqual(canonical_title("Sinfonia 5.mid"), "sinfonia5")
        self.assertEqual(
            canonical_title("For Your Precious Love_1.mid"),
            canonical_title("For Your Precious Love.mid"),
        )
        self.assertEqual(
            canonical_title("Maple Leaf Rag (LP Version).mid"),
            canonical_title("Maple Leaf Rag.mid"),
        )

    def test_distance_family_is_symmetric_for_known_curves(self) -> None:
        a = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        b = np.asarray([[0.0, 1.0], [1.0, 1.0]])
        hd, q95, mhd = nearest_distance_summaries(a, b)
        self.assertAlmostEqual(hd, 1.0)
        self.assertAlmostEqual(q95, 1.0)
        self.assertAlmostEqual(mhd, 1.0)

    def test_melody_fingerprint_ignores_midi_metadata(self) -> None:
        notes = (
            b"\x00\x90\x3c\x40"
            b"\x83\x60\x90\x3e\x50"
            b"\x83\x60\x90\x40\x60"
            b"\x00\xff\x2f\x00"
        )
        with_metadata = b"\x00\xff\x01\x03abc" + notes
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "plain.mid"
            tagged = Path(directory) / "tagged.mid"
            plain.write_bytes(midi_file(0, notes))
            tagged.write_bytes(midi_file(0, with_metadata))
            self.assertEqual(melody_fingerprint(plain), melody_fingerprint(tagged))

    def test_duplicate_union_marks_cross_genre_component(self) -> None:
        rows = [
            {
                "genre": "Classical",
                "file": "a.mid",
                "relative_path": "Classical/a.mid",
                "canonical_title": "piecea",
                "sha256": "same-bytes",
                "melody_fingerprint": "melody-a",
                "fingerprint_error": "",
            },
            {
                "genre": "Jazz",
                "file": "renamed.mid",
                "relative_path": "Jazz/renamed.mid",
                "canonical_title": "differentname",
                "sha256": "same-bytes",
                "melody_fingerprint": "melody-a",
                "fingerprint_error": "",
            },
        ]
        augmented, groups, summary = build_duplicate_groups(rows)
        self.assertEqual(len(groups), 1)
        self.assertTrue(all(row["cross_genre"] == 1 for row in augmented))
        self.assertEqual(summary["cross_genre_groups"], 1)
        self.assertEqual(summary["sha256_duplicate_sets"], 1)

    def test_multivariate_dtw_is_zero_and_symmetric(self) -> None:
        a = np.asarray([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        b = np.asarray([[0.0, 0.0], [1.5, 0.5], [2.0, 1.0]])
        self.assertAlmostEqual(_dtw_python(a, a, 1), 0.0)
        self.assertAlmostEqual(_dtw_python(a, b, 1), _dtw_python(b, a, 1))

    def test_distance_validation_rejects_nonzero_diagonal(self) -> None:
        matrix = np.asarray([[0.1, 1.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "diagonal"):
            validate_distance_matrix("bad", matrix)

    def test_knn_probabilities_are_normalized(self) -> None:
        labels = np.asarray(["Classical", "Jazz", "Rock", "Blues", "Electronic"])
        matrix = np.ones((5, 5)) - np.eye(5)
        probabilities = predict_knn_proba(
            matrix,
            np.asarray([0, 1, 2, 3]),
            np.asarray([4]),
            labels,
            k=3,
        )
        self.assertEqual(probabilities.shape, (1, 5))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities >= 0))
        grid = predict_knn_proba_candidates(
            matrix,
            np.asarray([0, 1, 2, 3]),
            np.asarray([4]),
            labels,
            candidates=(1, 3, 7),
        )
        np.testing.assert_allclose(grid[3], probabilities)
        self.assertAlmostEqual(float(grid[7].sum()), 1.0)

    def test_group_overlap_is_rejected(self) -> None:
        groups = np.asarray(["a", "b", "a"])
        with self.assertRaisesRegex(RuntimeError, "group leakage"):
            assert_group_separation(np.asarray([0, 1]), np.asarray([2]), groups)

    def test_sample_group_order_is_reproducible_and_seed_sensitive(self) -> None:
        groups = [f"group-{index:02d}" for index in range(20)]
        first = sampled_group_order(groups, 2026, "Classical")
        repeated = sampled_group_order(groups, 2026, "Classical")
        changed = sampled_group_order(groups, 2027, "Classical")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertEqual(set(first), set(groups))

    def test_validation_aggregation_and_paired_bootstrap(self) -> None:
        rows = []
        methods = (
            "Tuned MHD (nested)",
            "RF descriptors",
            "MHD-RF probability fusion",
        )
        for sample_seed in (2026, 2027, 2028):
            for outer_seed in (11, 23, 42, 67, 101):
                for method, accuracy in zip(methods, (0.40, 0.56, 0.54)):
                    rows.append(
                        {
                            "sample_seed": sample_seed,
                            "outer_seed": outer_seed,
                            "method": method,
                            "accuracy": accuracy,
                            "balanced_accuracy": accuracy,
                            "macro_f1": accuracy,
                            "fold_mean": accuracy,
                            "fold_std": 0.01,
                        }
                    )
        aggregate = aggregate_validation_runs(rows)
        self.assertEqual({row["repeats"] for row in aggregate}, {15})
        paired = paired_bootstrap_model_differences(
            rows, iterations=200, seed=2026
        )
        self.assertAlmostEqual(paired[0]["accuracy_difference_mean"], -0.16)
        self.assertAlmostEqual(paired[1]["accuracy_difference_mean"], 0.02)

    def test_method_figure_is_preserved_and_has_no_version_title(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "method_taxonomy_cn.png"
            prompt = root / "method_taxonomy_cn.prompt.txt"
            output = root / "output"
            Image.new("RGB", (2, 2), color="white").save(source)
            prompt.write_text("academic method diagram", encoding="utf-8")
            before = sha256_file(source)
            metadata = preserve_method_taxonomy(source, prompt, output)
            self.assertEqual(before, sha256_file(source))
            self.assertEqual(metadata["sha256"], before)
            self.assertTrue((output / source.name).exists())
        asset_source = Path(__file__).with_name("paper_assets.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"v3 ', asset_source)

    def test_feature_groups_cover_every_descriptor_once(self) -> None:
        melody = np.column_stack(
            (
                np.arange(40, dtype=float) * 0.5,
                60 + np.sin(np.arange(40)) * 5,
                70 + np.cos(np.arange(40)) * 10,
            )
        )
        values, names = melody_features(melody)
        groups = feature_group_indices(names)
        covered = np.concatenate(list(groups.values()))
        self.assertEqual(len(values), len(names))
        np.testing.assert_array_equal(np.sort(covered), np.arange(len(names)))

    def test_small_grouped_evaluation_runs_end_to_end(self) -> None:
        labels = np.repeat(
            np.asarray(["Classical", "Jazz", "Rock", "Blues", "Electronic"]), 5
        )
        groups = np.asarray([f"group-{index}" for index in range(len(labels))])
        coordinates = np.arange(len(labels), dtype=float)[:, None]
        matrix = np.abs(coordinates - coordinates.T)
        summary, predictions, probabilities, folds = evaluate_distance_metric(
            "synthetic",
            matrix,
            labels,
            groups,
            inner_seeds=(2026,),
            bootstrap_iterations=20,
        )
        self.assertEqual(len(predictions), len(labels))
        self.assertEqual(probabilities.shape, (len(labels), 5))
        self.assertEqual(len(folds), 5)
        self.assertIn("accuracy_ci_low", summary)


if __name__ == "__main__":
    unittest.main()
