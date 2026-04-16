#!/usr/bin/env python3
"""Crop head images from real_data videos into mask class directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "real_data"
DEFAULT_OUTPUT_DIR = ROOT / "real_data" / "train"
DEFAULT_FRAME_STEP = 1
MIN_DIMENSION = 6
MAX_DIMENSION = 750
DEFAULT_DETECTOR_MODEL = ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
DETECTOR_INPUT_SIZE = 640
HEAD_LABEL = 7
HEAD_THRESHOLD = 0.35
CLASS_DIR_MAP = {
    "no_masked": "0",
    "masked": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop head regions from no_masked*/masked* videos into class folders train/0 and train/1."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="Directory containing real_data videos.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root directory to store cropped images.")
    parser.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP, help="Take every Nth frame (default: 1).")
    parser.add_argument("--min-dimension", type=int, default=MIN_DIMENSION, help="Minimum crop width/height (default: 6).")
    parser.add_argument("--max-dimension", type=int, default=MAX_DIMENSION, help="Maximum crop width/height (default: 750).")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=DEFAULT_DETECTOR_MODEL,
        help="ONNX detector used to find head boxes (default: deimv2...640.onnx).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs if duplicates occur.")
    parser.add_argument("--dry-run", action="store_true", help="Plan operations without writing files.")
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1")
    if args.max_dimension < args.min_dimension:
        parser.error("--max-dimension must be greater than or equal to --min-dimension")
    return args


def load_detector_session(model_path: Path) -> tuple[ort.InferenceSession, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": ".",
                "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def _prepare_detector_blob(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        image,
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    blob = resized.transpose(2, 0, 1).astype(np.float32, copy=False)
    blob = np.expand_dims(blob, axis=0)
    return blob


def _run_detector(session: ort.InferenceSession, input_name: str, image: np.ndarray) -> np.ndarray:
    blob = _prepare_detector_blob(image)
    return session.run(None, {input_name: blob})[0][0]


def detect_head_box(
    session: ort.InferenceSession,
    input_name: str,
    frame: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    detections = _run_detector(session, input_name, frame)
    best_det = None
    best_score = HEAD_THRESHOLD
    for det in detections:
        label = int(round(det[0]))
        score = float(det[5])
        if label != HEAD_LABEL or score < best_score:
            continue
        best_score = score
        best_det = det
    if best_det is None:
        return None
    return float(best_det[1]), float(best_det[2]), float(best_det[3]), float(best_det[4])


def crop_frame_using_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> Optional[tuple[np.ndarray, int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    x1_px = max(int(round(x1 * width)), 0)
    y1_px = max(int(round(y1 * height)), 0)
    x2_px = min(int(round(x2 * width)), width)
    y2_px = min(int(round(y2 * height)), height)
    if x2_px <= x1_px or y2_px <= y1_px:
        return None
    crop = frame[y1_px:y2_px, x1_px:x2_px].copy()
    return crop, crop.shape[1], crop.shape[0]


def infer_class_name(stem: str) -> Optional[str]:
    lower = stem.lower()
    if lower.startswith("no_masked"):
        return "no_masked"
    if lower.startswith("masked"):
        return "masked"
    return None


def iter_video_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for path in sorted(input_dir.glob("*.mp*")):
        if infer_class_name(path.stem) is not None:
            yield path
        else:
            print(f"[skip] {path.name} (does not match no_masked*/masked*).", file=sys.stderr)


def save_frame(frame: np.ndarray, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    detector_session: ort.InferenceSession,
    detector_input_name: str,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")

    frame_index = 0
    saved = 0
    class_name = infer_class_name(video_path.stem)
    if class_name is None:
        raise RuntimeError(f"Unsupported video name: {video_path.name}")
    class_dir = CLASS_DIR_MAP[class_name]
    output_dir = args.output_dir / class_dir
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_index % args.frame_step != 0:
            frame_index += 1
            continue

        box = detect_head_box(detector_session, detector_input_name, frame)
        if box is None:
            frame_index += 1
            continue

        crop_result = crop_frame_using_box(frame, box)
        if crop_result is None:
            frame_index += 1
            continue
        crop, width_px, height_px = crop_result
        if (
            width_px < args.min_dimension
            or height_px < args.min_dimension
            or width_px > args.max_dimension
            or height_px > args.max_dimension
        ):
            frame_index += 1
            continue

        if class_name == "masked":
            filename = f"{video_path.stem}_mask_{frame_index:06d}.png"
        else:
            filename = f"{video_path.stem}_{frame_index:06d}.png"
        output_path = output_dir / filename
        if not args.dry_run:
            try:
                save_frame(crop, output_path, overwrite=args.overwrite)
            except FileExistsError:
                frame_index += 1
                continue
        saved += 1
        frame_index += 1

    capture.release()
    print(f"[info] Processed {video_path.name}: saved {saved} head crops.")
    return saved


def main() -> None:
    args = parse_args()
    detector_session, detector_input_name = load_detector_session(args.detector_model)

    total_saved = 0
    video_paths = list(iter_video_files(args.input_dir))
    if not video_paths:
        print("[info] No matching videos found.")
        return

    for video_path in video_paths:
        saved = process_video(video_path, args, detector_session, detector_input_name)
        total_saved += saved

    if args.dry_run:
        print("[dry-run] Skipped writing files.")
    print(f"[done] Saved {total_saved} crops from {len(video_paths)} videos.")


if __name__ == "__main__":
    main()
