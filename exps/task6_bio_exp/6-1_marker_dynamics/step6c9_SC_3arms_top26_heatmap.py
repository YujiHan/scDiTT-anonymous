from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from step6c8_SC_3arms_sorted_heatmap import render  # noqa: E402

DROP_BOTTOM = 9
STEM = "6-1-4g_heatmap_SC_combined_top26_3arms_sorted"
FONT_SCALE = 1.35
W_PER_COL = 3.4


def main() -> None:
    render(drop_bottom=DROP_BOTTOM, stem=STEM,
           font_scale=FONT_SCALE, w_per_col=W_PER_COL)


if __name__ == "__main__":
    main()
