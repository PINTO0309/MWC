#!/usr/bin/env python

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm


DEFAULT_ARCHIVE = Path("/path/to/train_aug_120x120_part_masked_clean.tar.gz")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_IMAGES_PER_DIR = 2000
DEFAULT_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract image files from a tar.gz archive into zero-padded output folders under data/, "
            "placing a fixed number of images in each folder."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Source tar.gz archive path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory where numbered folders will be created.",
    )
    parser.add_argument(
        "--images-per-dir",
        type=int,
        default=DEFAULT_IMAGES_PER_DIR,
        help="Maximum number of images to place in each numbered folder (default: 2000).",
    )
    parser.add_argument(
        "--start-dir-index",
        type=int,
        default=None,
        help="Optional starting folder index. Defaults to max existing numeric folder + 1, or 1.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=DEFAULT_EXTENSIONS,
        help="Image file extensions to include (default: .png .jpg .jpeg).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to extract.",
    )
    args = parser.parse_args()
    if args.images_per_dir < 1:
        parser.error("--images-per-dir must be at least 1")
    if args.start_dir_index is not None and args.start_dir_index < 1:
        parser.error("--start-dir-index must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def iter_image_members(archive_path: Path, extensions: Iterable[str]) -> Iterable[tarfile.TarInfo]:
    allowed = {ext.lower() for ext in extensions}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix in allowed:
                yield archive, member


def resolve_start_dir_index(output_dir: Path, explicit_index: int | None) -> int:
    if explicit_index is not None:
        return explicit_index

    highest = 0
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir() and child.name.isdigit() and len(child.name) == 6:
                highest = max(highest, int(child.name))
    return highest + 1 if highest > 0 else 1


def make_target_path(base_dir: Path, global_index: int, images_per_dir: int, filename: str) -> Path:
    folder_index = ((global_index - 1) // images_per_dir) + 1
    target_dir = base_dir / f"{folder_index:06d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / filename
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    return target_dir / f"{stem}_{global_index:08d}{suffix}"


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    images_per_dir: int,
    start_dir_index: int,
    extensions: Iterable[str],
    limit: int | None = None,
) -> tuple[int, int]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    last_folder_index = start_dir_index - 1
    start_global_index = ((start_dir_index - 1) * images_per_dir) + 1

    progress_total = limit if limit is not None else None
    with tqdm(
        total=progress_total,
        desc="Extracting",
        unit="img",
        dynamic_ncols=True,
    ) as progress:
        for archive, member in iter_image_members(archive_path, extensions):
            if limit is not None and extracted >= limit:
                break

            extracted += 1
            global_index = start_global_index + extracted - 1
            filename = Path(member.name).name
            target_path = make_target_path(output_dir, global_index, images_per_dir, filename)
            last_folder_index = int(target_path.parent.name)

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Failed to read member from archive: {member.name}")
            with source, target_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            progress.update(1)

    return extracted, last_folder_index


def main() -> None:
    args = parse_args()
    start_dir_index = resolve_start_dir_index(args.output_dir, args.start_dir_index)
    extracted, last_folder_index = extract_archive(
        archive_path=args.archive,
        output_dir=args.output_dir,
        images_per_dir=args.images_per_dir,
        start_dir_index=start_dir_index,
        extensions=args.extensions,
        limit=args.limit,
    )
    if extracted == 0:
        raise RuntimeError(f"No matching images found in {args.archive}")

    print(
        "Extracted "
        f"{extracted} images from {args.archive} into {args.output_dir} "
        f"(folders {start_dir_index:06d}..{last_folder_index:06d}, {args.images_per_dir} images per folder)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
