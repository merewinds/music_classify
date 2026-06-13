"""Core utilities for melody-curve geometry experiments.

This module intentionally avoids third-party MIDI parsers.  ADL files are
standard MIDI files, and the experiment only needs note-on events, so a small
parser keeps the final pipeline reproducible with the scientific Python stack.
"""

from __future__ import annotations

import math
import re
import struct
import unicodedata
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for index in range(4):
        if pos >= len(data):
            raise ValueError("truncated variable-length quantity")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, pos
        if index == 3:
            raise ValueError("variable-length quantity exceeds four bytes")
    raise AssertionError("unreachable")


def _parse_track(track: bytes) -> list[tuple[int, int, int]]:
    """Return (absolute_tick, pitch, velocity) note-on events."""
    notes: list[tuple[int, int, int]] = []
    pos = 0
    tick = 0
    running_status: int | None = None

    while pos < len(track):
        delta, pos = _read_vlq(track, pos)
        tick += delta
        if pos >= len(track):
            break

        first = track[pos]
        if first >= 0x80:
            status = first
            pos += 1
            if status < 0xF0:
                running_status = status
        elif running_status is not None:
            status = running_status
        else:
            raise ValueError("running status without previous channel status")

        if status == 0xFF:
            if pos >= len(track):
                raise ValueError("truncated meta event")
            pos += 1  # meta type
            length, pos = _read_vlq(track, pos)
            if pos + length > len(track):
                raise ValueError("truncated meta-event payload")
            pos += length
            running_status = None
            continue

        if status in (0xF0, 0xF7):
            length, pos = _read_vlq(track, pos)
            if pos + length > len(track):
                raise ValueError("truncated system-exclusive payload")
            pos += length
            running_status = None
            continue

        event_type = status & 0xF0
        data_len = 1 if event_type in (0xC0, 0xD0) else 2
        if pos + data_len > len(track):
            raise ValueError("truncated MIDI channel event")
        data1 = track[pos]
        data2 = track[pos + 1] if data_len == 2 else 0
        pos += data_len

        if event_type == 0x90 and data2 > 0:
            notes.append((tick, data1, data2))

    return notes


def parse_midi_note_ons(path: str | Path) -> tuple[np.ndarray, int]:
    """Parse a MIDI file into note-on events and ticks per quarter note.

    Returns
    -------
    notes:
        Array with columns (tick, pitch, velocity), sorted by tick.
    division:
        Ticks per quarter note.
    """
    data = Path(path).read_bytes()
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("not a standard MIDI file")

    header_len = struct.unpack(">I", data[4:8])[0]
    if header_len < 6:
        raise ValueError("invalid MIDI header")
    midi_format, n_tracks, division = struct.unpack(">HHH", data[8:14])
    if midi_format not in (0, 1):
        raise ValueError(f"MIDI format {midi_format} is not supported")
    if midi_format == 0 and n_tracks != 1:
        raise ValueError("MIDI format 0 must contain exactly one track")
    if division & 0x8000:
        raise ValueError("SMPTE timing is not supported")
    if division == 0:
        raise ValueError("ticks per quarter note must be positive")

    pos = 8 + header_len
    notes: list[tuple[int, int, int]] = []
    tracks_seen = 0
    while pos + 8 <= len(data) and tracks_seen < n_tracks:
        chunk_type = data[pos : pos + 4]
        chunk_len = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        if pos + 8 + chunk_len > len(data):
            raise ValueError("truncated MIDI track chunk")
        chunk = data[pos + 8 : pos + 8 + chunk_len]
        pos += 8 + chunk_len
        if chunk_type != b"MTrk":
            continue
        notes.extend(_parse_track(chunk))
        tracks_seen += 1

    if not notes:
        raise ValueError("MIDI file contains no note-on events")
    notes.sort(key=lambda x: (x[0], x[1], x[2]))
    return np.asarray(notes, dtype=np.float64), int(division)


def skyline_melody(note_ons: np.ndarray, division: int) -> np.ndarray:
    """Extract a highest-note skyline as (beat, pitch, velocity)."""
    melody = []
    i = 0
    while i < len(note_ons):
        tick = note_ons[i, 0]
        j = i + 1
        while j < len(note_ons) and note_ons[j, 0] == tick:
            j += 1
        chord = note_ons[i:j]
        top_pitch = chord[:, 1].max()
        top_notes = chord[chord[:, 1] == top_pitch]
        best = top_notes[np.argmax(top_notes[:, 2])]
        melody.append((tick / division, best[1], best[2]))
        i = j
    return np.asarray(melody, dtype=np.float64)


def resample_in_time(melody: np.ndarray, n_points: int = 96) -> np.ndarray:
    """Connect onset points linearly and sample uniformly in normalized time."""
    if len(melody) < 2:
        raise ValueError("melody must contain at least two onset points")
    time = melody[:, 0]
    keep = np.concatenate(([True], np.diff(time) > 0))
    melody = melody[keep]
    if len(melody) < 2 or melody[-1, 0] <= melody[0, 0]:
        raise ValueError("melody has zero duration")

    phase = (melody[:, 0] - melody[0, 0]) / (melody[-1, 0] - melody[0, 0])
    grid = np.linspace(0.0, 1.0, n_points)
    pitch = np.interp(grid, phase, melody[:, 1])
    velocity = np.interp(grid, phase, melody[:, 2])
    return np.column_stack((grid, pitch, velocity))


def local_minmax_curve(curve: np.ndarray) -> np.ndarray:
    result = np.empty_like(curve, dtype=np.float64)
    for col in range(curve.shape[1]):
        values = curve[:, col]
        span = values.max() - values.min()
        result[:, col] = (values - values.min()) / span if span > 0 else 0.5
    return result


def relative_curve(curve: np.ndarray, velocity_weight: float = 0.25) -> np.ndarray:
    """Use phase, transposition-invariant pitch, and physically scaled velocity."""
    pitch = (curve[:, 1] - np.median(curve[:, 1])) / 12.0
    velocity = velocity_weight * (curve[:, 2] - 64.0) / 32.0
    return np.column_stack((curve[:, 0], pitch, velocity))


def nearest_distance_summaries(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    tree_a: cKDTree | None = None,
    tree_b: cKDTree | None = None,
) -> tuple[float, float, float]:
    """Return max-Hausdorff, 95%-Hausdorff, and modified Hausdorff."""
    tree_a = tree_a or cKDTree(curve_a)
    tree_b = tree_b or cKDTree(curve_b)
    d_ab = tree_b.query(curve_a, workers=1)[0]
    d_ba = tree_a.query(curve_b, workers=1)[0]
    hd = max(float(d_ab.max()), float(d_ba.max()))
    q95 = max(float(np.quantile(d_ab, 0.95)), float(np.quantile(d_ba, 0.95)))
    mhd = max(float(d_ab.mean()), float(d_ba.mean()))
    return hd, q95, mhd


_VERSION_TERMS = (
    r"album|live|remaster(?:ed)?|version|mix|remix|take|edit|recording|"
    r"instrumental|karaoke|demo|mono|stereo|lp"
)
_VERSION_WORDS = re.compile(
    rf"\b(?:{_VERSION_TERMS})\b.*$",
    flags=re.IGNORECASE,
)
_BRACKETED_VERSION = re.compile(
    rf"[\(\[][^)\]]*\b(?:{_VERSION_TERMS})\b[^)\]]*[\)\]]",
    flags=re.IGNORECASE,
)


def canonical_title(path: str | Path) -> str:
    """Normalize obvious alternate-version suffixes for grouped validation."""
    text = unicodedata.normalize("NFKD", Path(path).stem).casefold()
    text = _BRACKETED_VERSION.sub(" ", text)
    text = _VERSION_WORDS.sub(" ", text)
    # Dataset duplicate exports commonly end in "_1" or "-2".  A plain
    # trailing number is retained because it may be a work number
    # (for example, "Sinfonia 5").
    text = re.sub(r"(?:[_-]\s*(?:take\s*)?\d+)$", " ", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text or f"untitled-{Path(path).name.casefold()}"


def melody_features(melody: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract interpretable non-geometric descriptors for an RF benchmark."""
    t = melody[:, 0]
    pitch = melody[:, 1]
    velocity = melody[:, 2]
    intervals = np.diff(pitch)
    ioi = np.diff(t)
    duration = max(float(t[-1] - t[0]), 1e-9)
    abs_intervals = np.abs(intervals)

    pc = np.mod(np.rint(pitch).astype(int), 12)
    pc_hist = np.bincount(pc, minlength=12).astype(float)
    pc_hist /= max(pc_hist.sum(), 1.0)
    nz = pc_hist[pc_hist > 0]
    pc_entropy = float(-(nz * np.log2(nz)).sum())

    signs = np.sign(intervals)
    direction_change = (
        np.mean(signs[1:] * signs[:-1] < 0) if len(signs) > 1 else 0.0
    )
    beat_fraction = np.mod(t, 1.0)
    offbeat = np.mean(np.minimum(beat_fraction, 1.0 - beat_fraction) > 0.10)
    phase = (t - t[0]) / duration
    pitch_centered = pitch - pitch.mean()
    contour_slope = float(np.polyfit(phase, pitch_centered, 1)[0])
    pitch_autocorrelation = (
        float(np.corrcoef(pitch_centered[:-1], pitch_centered[1:])[0, 1])
        if pitch.std() > 1e-9
        else 0.0
    )
    interval_bins = np.asarray([0, 1, 3, 5, 8, 13, np.inf])
    interval_classes = np.histogram(abs_intervals, bins=interval_bins)[0].astype(float)
    interval_classes /= max(interval_classes.sum(), 1.0)
    signed_intervals = np.clip(np.rint(intervals).astype(int), -12, 12) + 12
    interval_hist = np.bincount(signed_intervals, minlength=25).astype(float)
    interval_hist /= max(interval_hist.sum(), 1.0)
    interval_nz = interval_hist[interval_hist > 0]
    interval_entropy = float(-(interval_nz * np.log2(interval_nz)).sum())
    velocity_diff = np.diff(velocity)
    pitch_velocity_correlation = (
        float(np.corrcoef(pitch, velocity)[0, 1])
        if pitch.std() > 1e-9 and velocity.std() > 1e-9
        else 0.0
    )
    ioi_hist = np.histogram(
        np.log1p(ioi),
        bins=np.linspace(np.log1p(ioi).min(), np.log1p(ioi).max() + 1e-9, 9),
    )[0].astype(float)
    ioi_hist /= max(ioi_hist.sum(), 1.0)
    ioi_nz = ioi_hist[ioi_hist > 0]
    ioi_entropy = float(-(ioi_nz * np.log2(ioi_nz)).sum())

    names = [
        "duration_beats",
        "onset_count",
        "onset_density",
        "pitch_mean",
        "pitch_std",
        "pitch_range",
        "pitch_q10",
        "pitch_q90",
        "interval_abs_mean",
        "interval_abs_std",
        "interval_abs_max",
        "step_fraction",
        "leap_fraction",
        "repeat_fraction",
        "ascending_fraction",
        "descending_fraction",
        "direction_change",
        "velocity_mean",
        "velocity_std",
        "velocity_range",
        "ioi_mean",
        "ioi_std",
        "ioi_cv",
        "ioi_q25",
        "ioi_q75",
        "offbeat_fraction",
        "rhythm_regularity",
        "ioi_entropy",
        "contour_slope",
        "pitch_autocorrelation",
        "interval_entropy",
        "interval_unison_fraction",
        "interval_step_fraction",
        "interval_third_fourth_fraction",
        "interval_fifth_seventh_fraction",
        "interval_octave_fraction",
        "interval_large_fraction",
        "velocity_change_abs_mean",
        "velocity_change_std",
        "pitch_velocity_correlation",
        "pitch_class_entropy",
        "pitch_class_peak",
        "pitch_class_top3",
    ] + [f"pc_{i}" for i in range(12)]

    values = [
        duration,
        len(melody),
        len(melody) / duration,
        pitch.mean(),
        pitch.std(),
        np.ptp(pitch),
        np.quantile(pitch, 0.10),
        np.quantile(pitch, 0.90),
        abs_intervals.mean(),
        abs_intervals.std(),
        abs_intervals.max(),
        np.mean(abs_intervals <= 2),
        np.mean(abs_intervals >= 5),
        np.mean(abs_intervals == 0),
        np.mean(intervals > 0),
        np.mean(intervals < 0),
        direction_change,
        velocity.mean(),
        velocity.std(),
        np.ptp(velocity),
        ioi.mean(),
        ioi.std(),
        ioi.std() / max(ioi.mean(), 1e-9),
        np.quantile(ioi, 0.25),
        np.quantile(ioi, 0.75),
        offbeat,
        1.0 / (1.0 + ioi.std() / max(ioi.mean(), 1e-9)),
        ioi_entropy,
        contour_slope,
        pitch_autocorrelation,
        interval_entropy,
        *interval_classes.tolist(),
        np.mean(np.abs(velocity_diff)),
        velocity_diff.std(),
        pitch_velocity_correlation,
        pc_entropy,
        pc_hist.max(),
        np.sort(pc_hist)[-3:].sum(),
        *pc_hist.tolist(),
    ]
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    return values, names


def feature_group_indices(feature_names: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    """Map descriptor names to interpretable feature groups for ablation."""
    groups: dict[str, list[int]] = {
        "scale": [],
        "pitch": [],
        "interval_contour": [],
        "dynamics": [],
        "rhythm": [],
        "tonality": [],
    }
    for index, raw_name in enumerate(feature_names):
        name = str(raw_name)
        if name in {"duration_beats", "onset_count", "onset_density"}:
            groups["scale"].append(index)
        elif name.startswith("velocity") or name == "pitch_velocity_correlation":
            groups["dynamics"].append(index)
        elif name.startswith("ioi") or name in {
            "offbeat_fraction",
            "rhythm_regularity",
        }:
            groups["rhythm"].append(index)
        elif name.startswith("pc_") or name.startswith("pitch_class"):
            groups["tonality"].append(index)
        elif name.startswith("interval") or name in {
            "step_fraction",
            "leap_fraction",
            "repeat_fraction",
            "ascending_fraction",
            "descending_fraction",
            "direction_change",
            "contour_slope",
            "pitch_autocorrelation",
        }:
            groups["interval_contour"].append(index)
        else:
            groups["pitch"].append(index)
    return {
        name: np.asarray(indices, dtype=int)
        for name, indices in groups.items()
        if indices
    }


def load_one_midi(
    path: str | Path,
    n_points: int = 96,
    min_onsets: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    note_ons, division = parse_midi_note_ons(path)
    melody = skyline_melody(note_ons, division)
    if len(melody) < min_onsets:
        raise ValueError(f"too few melody onsets: {len(melody)}")
    curve = resample_in_time(melody, n_points=n_points)
    features, feature_names = melody_features(melody)
    metadata = {
        "onsets": float(len(melody)),
        "duration_beats": float(melody[-1, 0] - melody[0, 0]),
        "pitch_min": float(melody[:, 1].min()),
        "pitch_max": float(melody[:, 1].max()),
    }
    if not np.isfinite(curve).all() or not np.isfinite(features).all():
        raise ValueError("non-finite values after preprocessing")
    return curve, features, feature_names, metadata


def euclidean_point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(float(np.sum((a - b) ** 2)))
