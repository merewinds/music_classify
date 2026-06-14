"""Leakage-resistant dataset indexing, grouping, and balanced sampling."""

from __future__ import annotations

import csv
import hashlib
import struct
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from midi_geometry import (
    canonical_title,
    load_one_midi,
    parse_midi_note_ons,
    skyline_melody,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "adl-piano-midi"
RESULTS_DIR = ROOT / "results" / "final"
TABLE_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"
PIPELINE_VERSION = "v4"
GENRES = ("Classical", "Jazz", "Rock", "Blues", "Electronic")


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, TABLE_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dataset_cache_path(per_genre: int, n_points: int, seed: int) -> Path:
    return CACHE_DIR / (
        f"dataset_{PIPELINE_VERSION}_g{per_genre}_p{n_points}_s{seed}.npz"
    )


def audit_index_path() -> Path:
    return CACHE_DIR / f"dataset_index_{PIPELINE_VERSION}.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def melody_fingerprint(path: Path) -> str:
    """Hash normalized skyline geometry, ignoring metadata, tempo, and key."""
    note_ons, division = parse_midi_note_ons(path)
    melody = skyline_melody(note_ons, division)
    if len(melody) < 2:
        raise ValueError("too few skyline events for fingerprint")
    phase = (melody[:, 0] - melody[0, 0]) / max(
        float(melody[-1, 0] - melody[0, 0]), 1e-12
    )
    relative_pitch = melody[:, 1] - np.median(melody[:, 1])
    normalized = np.column_stack(
        (
            np.rint(phase * 10_000),
            np.rint(relative_pitch),
        )
    ).astype(">i4", copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray([len(melody)], dtype=">i4").tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _index_one(path: Path, genre: str) -> dict:
    row = {
        "genre": genre,
        "file": path.name,
        "relative_path": f"{genre}/{path.name}",
        "canonical_title": canonical_title(path),
        "file_size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": "",
        "melody_fingerprint": "",
        "fingerprint_error": "",
    }
    try:
        row["sha256"] = sha256_file(path)
        row["melody_fingerprint"] = melody_fingerprint(path)
    except (OSError, ValueError, struct.error) as exc:
        row["fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _current_file_signature() -> list[tuple[str, int, int]]:
    signature = []
    for genre in GENRES:
        for path in sorted((DATA_DIR / genre).glob("*.mid")):
            stat = path.stat()
            signature.append((f"{genre}/{path.name}", stat.st_size, stat.st_mtime_ns))
    return signature


def build_dataset_index(workers: int, force: bool = False) -> list[dict]:
    """Index all candidate files and cache content-based fingerprints."""
    ensure_dirs()
    cache_path = audit_index_path()
    signature = _current_file_signature()
    if cache_path.exists() and not force:
        cached = read_csv(cache_path)
        cached_signature = [
            (
                row["relative_path"],
                int(row["file_size"]),
                int(row["mtime_ns"]),
            )
            for row in cached
        ]
        if cached_signature == signature:
            print(f"[audit] loading dataset index: {cache_path}")
            return cached
        print("[audit] dataset files changed; rebuilding fingerprint index")

    tasks: list[tuple[Path, str]] = []
    for genre in GENRES:
        tasks.extend((path, genre) for path in sorted((DATA_DIR / genre).glob("*.mid")))

    rows: list[dict] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_index_one, path, genre): (path, genre)
            for path, genre in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 500 == 0 or completed == len(tasks):
                print(
                    f"[audit] indexed {completed}/{len(tasks)} files, "
                    f"{time.time() - started:.1f}s"
                )
    rows.sort(key=lambda row: (GENRES.index(row["genre"]), row["file"].casefold()))
    save_csv(cache_path, rows)
    return rows


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_duplicate_groups(index_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Merge title, byte-identical, and normalized-melody duplicate relations."""
    union = UnionFind(len(index_rows))
    identifier_maps: dict[str, dict[str, list[int]]] = {
        "title": defaultdict(list),
        "sha256": defaultdict(list),
        "melody": defaultdict(list),
    }
    for item, row in enumerate(index_rows):
        identifier_maps["title"][row["canonical_title"]].append(item)
        if row["sha256"]:
            identifier_maps["sha256"][row["sha256"]].append(item)
        if row["melody_fingerprint"]:
            identifier_maps["melody"][row["melody_fingerprint"]].append(item)

    for values in identifier_maps.values():
        for members in values.values():
            for item in members[1:]:
                union.union(members[0], item)

    components: dict[int, list[int]] = defaultdict(list)
    for item in range(len(index_rows)):
        components[union.find(item)].append(item)

    group_rows = []
    cross_genre_groups = set()
    component_id_by_item: dict[int, str] = {}
    for members in components.values():
        relative_paths = sorted(index_rows[item]["relative_path"] for item in members)
        component_id = "grp_" + hashlib.sha256(
            "\n".join(relative_paths).encode("utf-8")
        ).hexdigest()[:16]
        genres = sorted({index_rows[item]["genre"] for item in members})
        if len(genres) > 1:
            cross_genre_groups.add(component_id)
        for item in members:
            component_id_by_item[item] = component_id
        group_rows.append(
            {
                "group": component_id,
                "genres": "/".join(genres),
                "cross_genre": int(len(genres) > 1),
                "versions": len(members),
                "titles": "/".join(
                    sorted({index_rows[item]["canonical_title"] for item in members})
                ),
                "files": " | ".join(relative_paths),
            }
        )

    augmented = []
    for item, row in enumerate(index_rows):
        augmented.append(
            {
                **row,
                "group": component_id_by_item[item],
                "cross_genre": int(component_id_by_item[item] in cross_genre_groups),
            }
        )

    duplicate_counts = {}
    for name, values in identifier_maps.items():
        duplicate_counts[name] = sum(len(members) > 1 for members in values.values())
    summary = {
        "raw_files": len(index_rows),
        "fingerprint_failures": sum(bool(row["fingerprint_error"]) for row in index_rows),
        "title_duplicate_sets": duplicate_counts["title"],
        "sha256_duplicate_sets": duplicate_counts["sha256"],
        "melody_duplicate_sets": duplicate_counts["melody"],
        "union_groups": len(components),
        "cross_genre_groups": len(cross_genre_groups),
        "cross_genre_files": sum(
            row["group"] in cross_genre_groups for row in augmented
        ),
    }
    return augmented, sorted(group_rows, key=lambda row: row["group"]), summary


def sampling_rng(seed: int, genre: str) -> np.random.Generator:
    """Return a deterministic, genre-specific random generator."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), GENRES.index(genre)])
    )


def sampled_group_order(group_ids: list[str], seed: int, genre: str) -> list[str]:
    """Shuffle group identifiers reproducibly without coupling genres."""
    ordered = np.asarray(sorted(group_ids), dtype=object)
    rng = sampling_rng(seed, genre)
    return [str(value) for value in ordered[rng.permutation(len(ordered))]]


def _loaded_cache_is_valid(
    result: dict[str, np.ndarray], requested: int, n_points: int
) -> bool:
    required = {
        "curves",
        "features",
        "labels",
        "groups",
        "filenames",
        "feature_names",
        "file_sha256",
        "melody_fingerprints",
        "pipeline_version",
        "per_genre_actual",
    }
    if not required.issubset(result):
        return False
    actual = int(result["per_genre_actual"])
    return (
        str(result["pipeline_version"]) == PIPELINE_VERSION
        and actual <= requested
        and result["curves"].shape == (actual * len(GENRES), n_points, 3)
    )


def sample_dataset(
    requested_per_genre: int,
    n_points: int,
    seed: int,
    workers: int,
    force: bool = False,
    write_audit_tables: bool = True,
) -> dict[str, np.ndarray]:
    cache_path = dataset_cache_path(requested_per_genre, n_points, seed)
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=False)
        result = {key: loaded[key] for key in loaded.files}
        if _loaded_cache_is_valid(result, requested_per_genre, n_points):
            print(f"[data] loading balanced sample: {cache_path}")
            return result

    index_rows = build_dataset_index(workers=workers, force=force)
    indexed, group_rows, audit = build_duplicate_groups(index_rows)
    if write_audit_tables:
        save_csv(TABLE_DIR / "data_audit_files.csv", indexed)
        save_csv(TABLE_DIR / "data_audit_groups.csv", group_rows)
        conflict_groups = [row for row in group_rows if int(row["cross_genre"])]
        save_csv(TABLE_DIR / "cross_genre_conflicts.csv", conflict_groups)
        conflict_summary = []
        for genre_pair in sorted({row["genres"] for row in conflict_groups}):
            selected = [row for row in conflict_groups if row["genres"] == genre_pair]
            conflict_summary.append(
                {
                    "genres": genre_pair,
                    "group_count": len(selected),
                    "file_count": sum(int(row["versions"]) for row in selected),
                    "representative_title": selected[0]["titles"],
                    "representative_files": selected[0]["files"],
                }
            )
        save_csv(TABLE_DIR / "cross_genre_conflict_summary.csv", conflict_summary)

    selected_by_genre: dict[str, list[dict]] = {}
    failure_rows: list[dict] = []
    for genre in GENRES:
        rng = sampling_rng(seed, genre)
        candidates: dict[str, list[dict]] = defaultdict(list)
        for row in indexed:
            if row["genre"] == genre and not int(row["cross_genre"]):
                candidates[row["group"]].append(row)

        ordered_groups = sampled_group_order(list(candidates), seed, genre)
        selected = []
        failures = Counter()
        for group in ordered_groups:
            versions = candidates[group]
            version_order = rng.permutation(len(versions))
            parsed = None
            chosen = None
            for version_index in version_order:
                row = versions[int(version_index)]
                path = DATA_DIR / genre / row["file"]
                try:
                    parsed = load_one_midi(path, n_points=n_points)
                    chosen = row
                    break
                except (OSError, ValueError, struct.error) as exc:
                    failures[type(exc).__name__] += 1
                    failure_rows.append(
                        {
                            "genre": genre,
                            "file": row["file"],
                            "group": group,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            if parsed is None or chosen is None:
                continue
            curve, features, feature_names, metadata = parsed
            selected.append(
                {
                    "curve": curve,
                    "features": features,
                    "feature_names": feature_names,
                    "row": chosen,
                    "available_versions": len(versions),
                    "metadata": metadata,
                }
            )
            if len(selected) >= requested_per_genre:
                break
        selected_by_genre[genre] = selected
        print(
            f"[data] {genre}: {len(selected)}/{requested_per_genre} usable groups, "
            f"parse failures={sum(failures.values())}"
        )

    actual_per_genre = min(len(selected_by_genre[genre]) for genre in GENRES)
    if actual_per_genre == 0:
        raise RuntimeError("no balanced dataset can be formed after leakage filtering")
    if actual_per_genre < requested_per_genre:
        print(
            f"[data] target {requested_per_genre} unavailable; "
            f"using common balanced size {actual_per_genre}"
        )

    curves = []
    features = []
    labels = []
    groups = []
    filenames = []
    file_hashes = []
    melody_hashes = []
    metadata_rows = []
    feature_names: list[str] | None = None
    for genre in GENRES:
        for item in selected_by_genre[genre][:actual_per_genre]:
            row = item["row"]
            curves.append(item["curve"])
            features.append(item["features"])
            feature_names = item["feature_names"]
            labels.append(genre)
            groups.append(row["group"])
            filenames.append(row["file"])
            file_hashes.append(row["sha256"])
            melody_hashes.append(row["melody_fingerprint"])
            metadata_rows.append(
                {
                    "genre": genre,
                    "file": row["file"],
                    "group": row["group"],
                    "canonical_title": row["canonical_title"],
                    "available_versions": item["available_versions"],
                    "sha256": row["sha256"],
                    "melody_fingerprint": row["melody_fingerprint"],
                    **item["metadata"],
                }
            )

    audit_rows = []
    for genre in GENRES:
        raw = sum(row["genre"] == genre for row in indexed)
        cross = sum(
            row["genre"] == genre and int(row["cross_genre"]) for row in indexed
        )
        candidate_groups = len(
            {
                row["group"]
                for row in indexed
                if row["genre"] == genre and not int(row["cross_genre"])
            }
        )
        audit_rows.append(
            {
                "genre": genre,
                "raw_files": raw,
                "cross_genre_files_excluded": cross,
                "candidate_groups_after_dedup": candidate_groups,
                "usable_groups_seen": len(selected_by_genre[genre]),
                "selected_samples": actual_per_genre,
            }
        )
    audit_rows.append(
        {
            "genre": "TOTAL",
            **audit,
            "requested_per_genre": requested_per_genre,
            "actual_per_genre": actual_per_genre,
            "selected_samples": actual_per_genre * len(GENRES),
        }
    )
    if write_audit_tables:
        save_csv(TABLE_DIR / "data_audit_summary.csv", audit_rows)
        save_csv(TABLE_DIR / "sample_metadata.csv", metadata_rows)
        save_csv(TABLE_DIR / "sample_failures.csv", failure_rows)

    result = {
        "curves": np.asarray(curves, dtype=np.float64),
        "features": np.asarray(features, dtype=np.float64),
        "labels": np.asarray(labels, dtype=str),
        "groups": np.asarray(groups, dtype=str),
        "filenames": np.asarray(filenames, dtype=str),
        "feature_names": np.asarray(feature_names, dtype=str),
        "file_sha256": np.asarray(file_hashes, dtype=str),
        "melody_fingerprints": np.asarray(melody_hashes, dtype=str),
        "pipeline_version": np.asarray(PIPELINE_VERSION),
        "per_genre_requested": np.asarray(requested_per_genre),
        "per_genre_actual": np.asarray(actual_per_genre),
        "cross_genre_group_count": np.asarray(audit["cross_genre_groups"]),
    }
    if len(np.unique(result["groups"])) != len(result["groups"]):
        raise RuntimeError("balanced selection contains duplicate leakage groups")
    np.savez_compressed(cache_path, **result)
    print(f"[data] sample cache written: {cache_path}")
    return result
