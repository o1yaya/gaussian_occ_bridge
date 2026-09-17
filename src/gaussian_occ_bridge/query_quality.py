from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def audit_mask_vote_summary(
    summary: Mapping[str, Any],
    *,
    reject_dominant_view_share: float = 0.90,
    warn_dominant_view_share: float = 0.75,
    reject_min_vote_coverage: float = 0.50,
    warn_min_cross_view_support: int = 2,
) -> dict[str, Any]:
    """Audit an ILGS mask-vote summary without claiming semantic correctness.

    The automatic gate can reject clearly degenerate vote distributions, but a
    non-rejected result still requires visual review of query-mask and selected-ID
    overlays before it can be accepted.
    """
    if not 0.0 <= warn_dominant_view_share <= reject_dominant_view_share <= 1.0:
        raise ValueError("dominant-view thresholds must satisfy 0 <= warn <= reject <= 1")
    if not 0.0 <= reject_min_vote_coverage <= 1.0:
        raise ValueError("reject_min_vote_coverage must be in [0, 1]")
    if warn_min_cross_view_support < 1:
        raise ValueError("warn_min_cross_view_support must be positive")

    total_mask_pixels = int(summary.get("total_mask_pixels", 0))
    views = list(summary.get("views", []))
    selected_ids = [int(value) for value in summary.get("selected_ids", [])]
    if total_mask_pixels <= 0:
        raise ValueError("summary must contain a positive total_mask_pixels")
    if not views:
        raise ValueError("summary must contain at least one usable view")
    if not selected_ids:
        raise ValueError("summary must contain selected_ids")

    view_rows: list[dict[str, Any]] = []
    for view in views:
        pixels = int(view.get("mask_pixels", 0))
        if pixels < 0:
            raise ValueError("view mask_pixels must be non-negative")
        view_rows.append(
            {
                "index": int(view["index"]),
                "mask_pixels": pixels,
                "mask_share": pixels / total_mask_pixels,
            }
        )
    dominant_view = max(view_rows, key=lambda row: row["mask_share"])

    voted_pixels = sum(int(row.get("pixels", 0)) for row in summary.get("all_id_votes", []))
    vote_coverage = voted_pixels / total_mask_pixels
    selected_view_support: dict[str, list[int]] = {}
    for object_id in selected_ids:
        supporting_views = [
            int(view["index"])
            for view in views
            if int(view.get("ids", {}).get(str(object_id), 0)) > 0
        ]
        selected_view_support[str(object_id)] = supporting_views
    under_supported_ids = [
        int(object_id)
        for object_id, supporting_views in selected_view_support.items()
        if len(supporting_views) < warn_min_cross_view_support
    ]

    reject_reasons: list[str] = []
    warnings: list[str] = []
    if dominant_view["mask_share"] >= reject_dominant_view_share:
        reject_reasons.append("one view dominates the aggregate query mask")
    elif dominant_view["mask_share"] >= warn_dominant_view_share:
        warnings.append("one view contributes most aggregate query-mask pixels")
    if vote_coverage < reject_min_vote_coverage:
        reject_reasons.append("too few query-mask pixels retain object-ID votes")
    if under_supported_ids:
        warnings.append("some selected object IDs have weak cross-view support")

    return {
        "automatic_decision": "reject" if reject_reasons else "needs-visual-review",
        "reject_reasons": reject_reasons,
        "warnings": warnings,
        "usable_view_count": len(views),
        "total_mask_pixels": total_mask_pixels,
        "voted_mask_pixels": voted_pixels,
        "vote_coverage": vote_coverage,
        "dominant_view": dominant_view,
        "views": view_rows,
        "selected_ids": selected_ids,
        "selected_id_view_support": selected_view_support,
        "under_supported_selected_ids": under_supported_ids,
        "thresholds": {
            "reject_dominant_view_share": reject_dominant_view_share,
            "warn_dominant_view_share": warn_dominant_view_share,
            "reject_min_vote_coverage": reject_min_vote_coverage,
            "warn_min_cross_view_support": warn_min_cross_view_support,
        },
        "interpretation": (
            "Automatic checks detect degenerate vote distributions only. A non-rejected "
            "result still requires visual review before semantic acceptance."
        ),
    }
