"""Focused regression tests for the final melody-geometry pipeline."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from .midi_geometry import (
        _read_vlq,
        canonical_title,
        nearest_distance_summaries,
        parse_midi_note_ons,
        skyline_melody,
    )
except ImportError:
    from midi_geometry import (
        _read_vlq,
        canonical_title,
        nearest_distance_summaries,
        parse_midi_note_ons,
        skyline_melody,
    )


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


if __name__ == "__main__":
    unittest.main()
