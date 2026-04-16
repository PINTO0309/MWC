#!/usr/bin/env python

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


# Classes are fixed: 0 = no_masked, 1 = masked.
CLASS_MAP = {
    "no_masked": 0,
    "masked": 1,
}

SPLIT_MAP = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}

RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class SampleRow:
    split: str
    image_path: str
    image_bytes: bytes
    class_id: int
    label: str
    source: str
    filename: str
    video_id: str
    timestamp: int
    person_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dataset.parquet for the MWC mask classifier from numbered image folders under data/, "
            "embedding resized image bytes and deriving classes from filenames."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Root directory containing extracted images, typically in six-digit subfolders such as data/000001/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset.parquet"),
        help="Destination parquet file path.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".png", ".jpg", ".jpeg"],
        help="Image file extensions to include (default: .png .jpg .jpeg).",
    )
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default=parse_image_size("48x48"),
        metavar="HEIGHTxWIDTH",
        help="Resize images before embedding into parquet using HEIGHTxWIDTH format (default: 48x48).",
    )
    return parser.parse_args()


def parse_image_size(value: str) -> tuple[int, int]:
    normalized = value.strip().lower()
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid image size {value!r}; expected HEIGHTxWIDTH.")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid image size {value!r}; expected HEIGHTxWIDTH.") from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError(f"Invalid image size {value!r}; dimensions must be positive.")
    return width, height


def iter_images(base: Path, extensions: Iterable[str]) -> Iterable[tuple[Path, str]]:
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            rel_path = path.relative_to(base).as_posix()
            yield path, rel_path


def infer_split(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if parts:
        candidate = SPLIT_MAP.get(parts[0].lower())
        if candidate is not None:
            return candidate
    return "train"


def infer_label(filename: str) -> tuple[int, str]:
    if "_mask_" in filename.lower():
        return CLASS_MAP["masked"], "masked"
    return CLASS_MAP["no_masked"], "no_masked"


def encode_resized_image(path: Path, image_size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    format_name = "PNG"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        format_name = "JPEG"
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(image_size, RESAMPLING.LANCZOS)
        save_kwargs = {"format": format_name}
        if format_name == "JPEG":
            save_kwargs["quality"] = 95
        resized.save(output, **save_kwargs)
    return output.getvalue()


def build_dataframe(root: Path, extensions: Iterable[str], image_size: tuple[int, int]) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    image_entries = list(iter_images(root, extensions))
    if not image_entries:
        raise RuntimeError(f"No images found under {root}.")

    rows: List[SampleRow] = []
    person_counter = 1

    for path, rel_path in tqdm(
        image_entries,
        desc="Building parquet",
        unit="img",
        dynamic_ncols=True,
    ):
        split = infer_split(rel_path)
        class_id, label = infer_label(path.name)
        rel_parent = Path(rel_path).parent
        parent = rel_parent.as_posix()
        source = root.name
        if rel_parent != Path(".") and rel_parent.parts[0].isdigit() and len(rel_parent.parts[0]) == 6:
            source = rel_parent.parts[0]
        rows.append(
            SampleRow(
                split=split,
                image_path=rel_path,
                image_bytes=encode_resized_image(path, image_size),
                class_id=class_id,
                label=label,
                source=source,
                filename=path.name,
                video_id=root.name if parent == "." else parent,
                timestamp=0,
                person_id=person_counter,
            )
        )
        person_counter += 1

    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "class_id", "image_path"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    df = build_dataframe(args.root, args.extensions, args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} with resized image bytes {args.image_size[0]}x{args.image_size[1]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
