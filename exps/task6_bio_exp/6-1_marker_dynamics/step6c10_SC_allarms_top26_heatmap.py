from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from step6b_allmarker_figures import HEAT_ARMS
from step6c8_SC_3arms_sorted_heatmap import render  # noqa: E402

DROP_BOTTOM = 9
STEM = "6-1-4h_heatmap_SC_combined_top26_allarms_sorted"


def main() -> None:
    render(drop_bottom=DROP_BOTTOM, stem=STEM, display_arms=HEAT_ARMS)


if __name__ == "__main__":
    main()
