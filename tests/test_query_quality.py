import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gaussian_occ_bridge import audit_mask_vote_summary


def _summary(view_pixels, selected_ids=(7,)):
    views = []
    for index, pixels in enumerate(view_pixels):
        views.append(
            {
                "index": index,
                "mask_pixels": pixels,
                "ids": {str(object_id): pixels // len(selected_ids) for object_id in selected_ids},
            }
        )
    total = sum(view_pixels)
    return {
        "selected_ids": list(selected_ids),
        "all_id_votes": [{"id": selected_ids[0], "pixels": total, "ratio": 1.0}],
        "views": views,
        "total_mask_pixels": total,
    }


class QueryQualityTests(unittest.TestCase):
    def test_balanced_views_require_visual_review(self):
        result = audit_mask_vote_summary(_summary([100, 120, 80, 100]))
        self.assertEqual(result["automatic_decision"], "needs-visual-review")
        self.assertEqual(result["reject_reasons"], [])

    def test_dominant_view_is_rejected(self):
        result = audit_mask_vote_summary(_summary([10, 10, 10, 970]))
        self.assertEqual(result["automatic_decision"], "reject")
        self.assertIn("one view dominates the aggregate query mask", result["reject_reasons"])

    def test_low_vote_coverage_is_rejected(self):
        summary = _summary([100, 100, 100, 100])
        summary["all_id_votes"][0]["pixels"] = 100
        result = audit_mask_vote_summary(summary)
        self.assertEqual(result["automatic_decision"], "reject")
        self.assertAlmostEqual(result["vote_coverage"], 0.25)

    def test_single_view_id_is_warned(self):
        summary = _summary([100, 100, 100, 100], selected_ids=(7, 9))
        for view in summary["views"][1:]:
            view["ids"].pop("9")
        result = audit_mask_vote_summary(summary)
        self.assertIn(9, result["under_supported_selected_ids"])


if __name__ == "__main__":
    unittest.main()
