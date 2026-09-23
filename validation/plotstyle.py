"""Shared figure style: native matplotlib with seaborn's theme on top.

Every figure script imports its colours from here and calls ``apply()`` once. Colours are
matplotlib's default cycle (C0, C1, ...) and plain greys; there are no custom palettes or rcParams
blocks. seaborn (the ``plot`` extra) supplies the theme; without it the figures fall back to
matplotlib's defaults.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

# default colour cycle and greys (names kept short for use in the figure scripts)
S1, S2, S3, S4, S5 = "C0", "C1", "C2", "C3", "C4"
INK, INK2, MUTED, AXIS, SHADE, SURFACE = "black", "0.25", "0.45", "0.6", "0.92", "white"
THEORY = {"color": INK, "lw": 1.0}
LIMIT = {"color": MUTED, "lw": 1.0, "ls": "--"}
SEQUENTIAL = "viridis"  # matplotlib's default sequential colormap


def apply() -> None:
    plt.style.use("default")
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")  # native font
    plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight"})
