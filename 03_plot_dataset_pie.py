#!/usr/bin/env python3
"""Generate a class-ratio pie chart PNG from a parquet dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pie chart PNG for class composition in a parquet dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_48x48.parquet"),
        help="Input parquet path.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column used to compute composition ratio.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_48x48_class_ratio.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def format_autopct(values: list[int]):
    total = sum(values)

    def _formatter(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count:,})"

    return _formatter


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    table = pq.read_table(args.input, columns=[args.label_column])
    counts = table.column(args.label_column).to_pandas().value_counts().sort_index()
    if counts.empty:
        raise RuntimeError(f"No values found in column: {args.label_column}")

    labels = counts.index.tolist()
    values = counts.astype(int).tolist()
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=format_autopct(values),
        startangle=90,
        counterclock=False,
        colors=colors[: len(values)],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 11},
    )
    ax.set_title(
        f"dataset_48x48.parquet class ratio\nTotal: {total:,} samples",
        fontsize=14,
        pad=18,
    )
    ax.axis("equal")

    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(10)
        text.set_weight("bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote pie chart to {args.output}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
