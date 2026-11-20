#!/usr/bin/env python3
"""Calculate bacterial Hi-C directional-preference (DI) scores.

This command-line script is a cleaned, self-contained conversion of
``DI_analysis.ipynb``.  It reproduces the notebook's DI workflow:

1. ICE normalization (51 iterations; low-coverage threshold 100),
2. one round of sequential component normalization (SCN),
3. Pearson correlation between matrix rows, and
4. a paired t-test comparing log-transformed right- and left-contact vectors.

With 5-kb bins, ``--scales-kb 100 400`` uses 20 or 80 bins on *each* side of
the focal bin.  Circular wrap-around is enabled by default, as in the Lioy
``directional_indice.py`` implementation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_square_matrix(path: Path) -> np.ndarray:
    """Load a dense, whitespace-delimited square contact matrix."""
    matrix = np.loadtxt(path, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Input must be a dense square matrix; observed shape={matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Input matrix contains NaN or infinite values")
    if np.any(matrix < 0):
        raise ValueError("Input contact matrix contains negative values")
    return matrix


def ice_normalize(
    matrix: np.ndarray, threshold: float = 100.0, iterations: int = 51
) -> np.ndarray:
    """Reproduce ``ice_mirny3.ice_func`` used in the notebook.

    Bins whose initial column sum is below ``threshold`` are excluded from all
    normalization iterations and remain zero.
    """
    normalized = np.asarray(matrix, dtype=float).copy()
    initial_coverage = normalized.sum(axis=0)
    excluded = initial_coverage < threshold

    for _ in range(iterations):
        row_sums = normalized.sum(axis=1)
        col_sums = normalized.sum(axis=0)
        denominator = row_sums[:, None] * col_sums[None, :]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            normalized = np.divide(
                normalized,
                denominator,
                out=np.zeros_like(normalized),
                where=denominator != 0,
            )
        normalized[~np.isfinite(normalized)] = 0.0
        normalized[excluded, :] = 0.0
        normalized[:, excluded] = 0.0

    return normalized


def scn_normalize(
    matrix: np.ndarray, threshold: float = 0.0, iterations: int = 1
) -> np.ndarray:
    """Sequentially normalize rows and columns as in ``scn.scn_func``."""
    normalized = np.asarray(matrix, dtype=float).copy()
    keep = normalized.sum(axis=1) > threshold

    for _ in range(iterations):
        row_sums = normalized.sum(axis=1)
        row_normalized = np.divide(
            normalized,
            row_sums[:, None],
            out=np.zeros_like(normalized),
            where=(row_sums[:, None] != 0) & keep[:, None],
        )

        col_sums = row_normalized.sum(axis=0)
        normalized = np.divide(
            row_normalized,
            col_sums[None, :],
            out=np.zeros_like(row_normalized),
            where=(col_sums[None, :] != 0) & keep[None, :],
        )

    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Calculate the row-wise Pearson correlation matrix."""
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(matrix)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def _neighbor_indices(
    index: int, n_bins: int, window_bins: int, circular: bool
) -> Tuple[np.ndarray, np.ndarray]:
    left = np.arange(index - 1, index - window_bins - 1, -1)
    right = np.arange(index + 1, index + window_bins + 1)
    if circular:
        return left % n_bins, right % n_bins
    return left[(left >= 0) & (left < n_bins)], right[
        (right >= 0) & (right < n_bins)
    ]


def directional_preference(
    corr: np.ndarray, window_bins: int, circular: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired-t DI statistics and two-sided p-values for every bin.

    Positive correlation values are natural-log transformed.  Zero and
    negative values are replaced by zero, matching the original Lioy code.
    The test is ``ttest_rel(right, left)``, so positive DI indicates a
    downstream/rightward interaction preference and negative DI indicates an
    upstream/leftward preference.
    """
    n_bins = corr.shape[0]
    di = np.zeros(n_bins, dtype=float)
    p_value = np.ones(n_bins, dtype=float)

    if window_bins < 1:
        return di, p_value
    if circular and window_bins >= n_bins:
        raise ValueError("For a circular matrix, window_bins must be < n_bins")

    for i in range(n_bins):
        left_idx, right_idx = _neighbor_indices(
            i, n_bins, window_bins, circular=circular
        )
        if len(left_idx) != len(right_idx) or len(left_idx) < 2:
            continue

        left_raw = corr[i, left_idx]
        right_raw = corr[i, right_idx]
        left = np.zeros_like(left_raw)
        right = np.zeros_like(right_raw)
        left_positive = left_raw > 0
        right_positive = right_raw > 0
        left[left_positive] = np.log(left_raw[left_positive])
        right[right_positive] = np.log(right_raw[right_positive])

        if np.sum(left) == 0 or np.sum(right) == 0:
            continue

        result = stats.ttest_rel(right, left, nan_policy="omit")
        if np.isfinite(result.statistic):
            di[i] = float(result.statistic)
            p_value[i] = float(result.pvalue)

    return di, p_value


def parse_scales(scales_kb: Iterable[int], bin_size: int) -> Dict[int, int]:
    """Map requested scales in kb to exact numbers of bins per side."""
    parsed: Dict[int, int] = {}
    for scale_kb in scales_kb:
        scale_bp = int(scale_kb) * 1000
        if scale_bp % bin_size:
            raise ValueError(
                f"Scale {scale_kb} kb is not an integer multiple of "
                f"bin size {bin_size} bp"
            )
        parsed[int(scale_kb)] = scale_bp // bin_size
    return parsed


def write_di_table(
    path: Path,
    n_bins: int,
    bin_size: int,
    scores: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Write bin coordinates, DI statistics and p-values to CSV."""
    fieldnames = ["bin_index", "start_bp", "end_bp"]
    for scale_kb in scores:
        fieldnames.extend([f"DI_{scale_kb}kb", f"pvalue_{scale_kb}kb"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_bins):
            row = {
                "bin_index": i,
                "start_bp": i * bin_size,
                "end_bp": (i + 1) * bin_size,
            }
            for scale_kb, (di, p_value) in scores.items():
                row[f"DI_{scale_kb}kb"] = f"{di[i]:.10g}"
                row[f"pvalue_{scale_kb}kb"] = f"{p_value[i]:.10g}"
            writer.writerow(row)


def plot_di(
    path: Path,
    scores: Dict[int, Tuple[np.ndarray, np.ndarray]],
    bin_size: int,
    title: str,
) -> None:
    """Plot positive DI in red and negative DI in green."""
    n_panels = len(scores)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(10, max(2.4 * n_panels, 3.2)),
        sharex=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (scale_kb, (di, _)) in zip(axes, scores.items()):
        x_mb = np.arange(di.size) * bin_size / 1_000_000
        ax.fill_between(x_mb, 0, di, where=di >= 0, color="#C43C39", linewidth=0)
        ax.fill_between(x_mb, 0, di, where=di < 0, color="#2F8F65", linewidth=0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylim(-2, 2)
        ax.set_ylabel(f"DI\n({scale_kb} kb)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Genome position (Mb)")
    fig.savefig(path.with_suffix(".pdf"), dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate Lioy-style directional-preference scores."
    )
    parser.add_argument("input_matrix", type=Path, help="Dense square contact matrix")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("DI_analysis"),
        help="Output path prefix (default: ./DI_analysis)",
    )
    parser.add_argument("--bin-size", type=int, default=5000, help="Bin size in bp")
    parser.add_argument(
        "--scales-kb",
        type=int,
        nargs="+",
        default=[100, 400],
        help="Directional windows in kb on each side (default: 100 400)",
    )
    parser.add_argument(
        "--ice-threshold", type=float, default=100.0, help="Initial coverage threshold"
    )
    parser.add_argument("--ice-iterations", type=int, default=51)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip ICE and SCN if the matrix is already normalized",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Disable circular wrap-around at chromosome ends",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save normalized and correlation matrices as .npy files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be positive")
    if args.ice_iterations < 1:
        raise ValueError("--ice-iterations must be at least 1")

    scale_bins = parse_scales(args.scales_kb, args.bin_size)
    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    matrix = load_square_matrix(args.input_matrix.expanduser().resolve())
    if args.no_normalize:
        normalized = matrix.copy()
    else:
        normalized = ice_normalize(
            matrix,
            threshold=args.ice_threshold,
            iterations=args.ice_iterations,
        )
        normalized = scn_normalize(normalized, threshold=0.0, iterations=1)

    corr = correlation_matrix(normalized)
    scores: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for scale_kb, window_bins in scale_bins.items():
        scores[scale_kb] = directional_preference(
            corr, window_bins=window_bins, circular=not args.linear
        )

    csv_path = Path(f"{output_prefix}_DI.csv")
    write_di_table(csv_path, matrix.shape[0], args.bin_size, scores)
    plot_di(
        Path(f"{output_prefix}_DI_plot"),
        scores,
        args.bin_size,
        title=output_prefix.name,
    )

    if args.save_intermediates:
        np.save(f"{output_prefix}_normalized.npy", normalized)
        np.save(f"{output_prefix}_correlation.npy", corr)

    print(f"DI table: {csv_path}")
    print(f"DI plots: {output_prefix}_DI_plot.pdf and .png")


if __name__ == "__main__":
    main()
