from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_occ_bridge import audit_mask_vote_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an automatic quality gate to an ILGS mask-vote summary."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--reject-dominant-view-share", type=float, default=0.90)
    parser.add_argument("--warn-dominant-view-share", type=float, default=0.75)
    parser.add_argument("--reject-min-vote-coverage", type=float, default=0.50)
    parser.add_argument("--warn-min-cross-view-support", type=int, default=2)
    parser.add_argument(
        "--visual-decision", choices=["pending", "accept", "reject"], default="pending"
    )
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    source = Path(args.input)
    summary = json.loads(source.read_text(encoding="utf-8"))
    automatic = audit_mask_vote_summary(
        summary,
        reject_dominant_view_share=args.reject_dominant_view_share,
        warn_dominant_view_share=args.warn_dominant_view_share,
        reject_min_vote_coverage=args.reject_min_vote_coverage,
        warn_min_cross_view_support=args.warn_min_cross_view_support,
    )
    if automatic["automatic_decision"] == "reject" or args.visual_decision == "reject":
        final_decision = "reject"
    elif args.visual_decision == "accept":
        final_decision = "accept"
    else:
        final_decision = "needs-visual-review"

    output = {
        "query": args.query,
        "source_summary": source.name,
        **automatic,
        "visual_decision": args.visual_decision,
        "visual_note": args.visual_note,
        "final_decision": final_decision,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
