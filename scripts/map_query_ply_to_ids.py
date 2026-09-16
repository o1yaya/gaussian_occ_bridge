from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import load_ilgs_ply_hard_labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map a query-isolated ILGS Gaussian PLY to classifier object IDs."
    )
    parser.add_argument("--ply", required=True)
    parser.add_argument("--classifier-npz", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-ratio", type=float, default=0.01)
    args = parser.parse_args()
    if not 0.0 <= args.min_ratio <= 1.0:
        raise ValueError("min-ratio must be in [0, 1]")

    ply_path = Path(args.ply)
    classifier_path = Path(args.classifier_npz)
    batch = load_ilgs_ply_hard_labels(ply_path, classifier_path)
    object_ids, counts = np.unique(batch.semantic_ids, return_counts=True)
    total = int(counts.sum())
    rows = [
        {
            "object_id": int(object_id),
            "gaussian_count": int(count),
            "ratio": float(count / total),
        }
        for object_id, count in sorted(
            zip(object_ids, counts), key=lambda item: item[1], reverse=True
        )
    ]
    selected = [row["object_id"] for row in rows if row["ratio"] >= args.min_ratio]
    output = {
        "scene": args.scene,
        "query": args.query,
        "status": "mapped-from-query-isolated-ply",
        "object_ids": selected,
        "gaussian_count": total,
        "object_id_histogram": rows,
        "min_ratio": float(args.min_ratio),
        "source_ply": ply_path.name,
        "source_ply_sha256": _sha256(ply_path),
        "classifier_npz": classifier_path.name,
        "classifier_npz_sha256": _sha256(classifier_path),
        "interpretation": (
            "All mapped IDs belong to the exported query subset. This mapping does not "
            "assign human-readable names to any other classifier IDs."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

