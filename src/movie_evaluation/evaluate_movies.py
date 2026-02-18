#!/usr/bin/env python
# coding: utf-8

"""
Standalone movie evaluation script.
Evaluates trained model(s) on movies and saves per-frame probability vectors
for downstream ensemble analysis.

Usage:
    # Single model (output auto-derived: evaluation/<model_name>/movies/)
    python src/movie_evaluation/evaluate_movies.py --checkpoint models/pixabay_light_v2/model.pt

    # Batch mode — multiple checkpoints
    python src/movie_evaluation/evaluate_movies.py --checkpoint models/pixabay_light_v2/model.pt models/pixabay_heavy_v1/model.pt

    # Batch mode — all model.pt files under a directory
    python src/movie_evaluation/evaluate_movies.py --checkpoint-dir models/

    # Override defaults
    python src/movie_evaluation/evaluate_movies.py --checkpoint models/pixabay_light_v2/model.pt --frame-stride 50 --device cpu
"""

from pathlib import Path
import json
from typing import List, Dict, Optional

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import polars as pl
import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm.auto import tqdm


# -------------------------
# Default Paths (convention-based)
# -------------------------

MOVIES_DIR = Path.home() / "Movies"
MOVIE_LIST = Path("src/movie_evaluation/movie_list.txt")
FACE_DETECTOR = Path("models/mediapipe_face_detector/detector.tflite")
EVALUATION_ROOT = Path("evaluation")


# -------------------------
# Model Loading
# -------------------------

def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """
    Load model from checkpoint file.

    Returns:
        model: Loaded model
        emotions: List of emotion labels
        transform: Transform to use for evaluation
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    emotions = checkpoint["emotions"]
    num_classes = len(emotions)

    # Get config if available (for new checkpoints)
    config = checkpoint.get("config", {})

    # Rebuild model
    model_arch = config.get("model", {}).get("architecture", "resnet18")

    if model_arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
    elif model_arch == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown architecture: {model_arch}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Create evaluation transform
    image_size = config.get("data", {}).get("image_size", 224)
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, emotions, eval_transform


def derive_model_name(checkpoint_path: Path) -> str:
    """
    Derive a model name from the checkpoint path.

    models/pixabay_light_v2/model.pt  -> pixabay_light_v2
    models/my_model/best.pt           -> my_model
    /abs/path/to/cool_model/model.pt  -> cool_model
    """
    return checkpoint_path.resolve().parent.name


# -------------------------
# Movie Processing
# -------------------------

def get_largest_detection(detections):
    """Get largest face from detections."""
    if not detections.detections:
        return None

    largest = None
    max_area = 0

    for detection in detections.detections:
        bbox = detection.bounding_box
        area = bbox.width * bbox.height
        if area > max_area:
            max_area = area
            largest = detection

    return largest


def crop_face(frame, bbox):
    """Crop face from frame using bounding box."""
    h, w = frame.shape[:2]
    x_min = max(0, int(bbox.origin_x))
    y_min = max(0, int(bbox.origin_y))
    x_max = min(w, int(bbox.origin_x + bbox.width))
    y_max = min(h, int(bbox.origin_y + bbox.height))

    return frame[y_min:y_max, x_min:x_max]


def process_movie(
    movie_path: Path,
    model,
    emotions: List[str],
    eval_transform,
    face_detector,
    device: str,
    frame_stride: int = 100,
    min_face_confidence: float = 0.5,
) -> pl.DataFrame:
    """
    Process a movie and return emotion predictions with full probability
    vectors for every detected face.
    """
    cap = cv2.VideoCapture(str(movie_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {movie_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=movie_path.stem, unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        # Only process every Nth frame
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect faces
        detections = face_detector.detect(mp_image)

        # Get largest face
        detection = get_largest_detection(detections)

        if detection is not None and detection.categories[0].score >= min_face_confidence:
            bbox = detection.bounding_box

            # Crop face
            face_crop = crop_face(rgb_frame, bbox)

            if face_crop.size > 0:
                # Convert to PIL and apply transform
                face_pil = Image.fromarray(face_crop)
                face_tensor = eval_transform(face_pil).unsqueeze(0).to(device)

                # Predict emotion — store full probability vector
                with torch.no_grad():
                    logits = model(face_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred = int(probs.argmax())
                    confidence = float(probs[pred])

                # Calculate face statistics
                face_area = bbox.width * bbox.height
                frame_area = frame.shape[0] * frame.shape[1]
                relative_face_area = face_area / frame_area

                row = {
                    "movie": movie_path.stem,
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "face_area": int(face_area),
                    "relative_face_area": relative_face_area,
                    "emotion": emotions[pred],
                    "confidence": confidence,
                }

                # Add per-emotion probability columns: prob_happy, prob_sad, ...
                for i, emo in enumerate(emotions):
                    row[f"prob_{emo}"] = float(probs[i])

                results.append(row)

        frame_idx += 1

    cap.release()
    pbar.close()

    return pl.DataFrame(results)


def evaluate_movies(
    movie_paths: List[Path],
    model,
    emotions: List[str],
    eval_transform,
    device: str,
    output_dir: Path,
    face_detector_path: Path,
    frame_stride: int = 100,
    min_face_confidence: float = 0.5,
) -> Dict:
    """
    Evaluate model on multiple movies.

    Returns summary statistics and the combined DataFrame.
    """
    # Initialize face detector
    print("Initializing face detector...")
    base_options = mp_python.BaseOptions(model_asset_path=str(face_detector_path))
    face_options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_face_confidence,
    )
    face_detector = vision.FaceDetector.create_from_options(face_options)

    # Process each movie
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for movie_path in movie_paths:
        print(f"\nProcessing: {movie_path.name}")

        out_path = output_dir / f"{movie_path.stem}.parquet"
        if out_path.exists():
            print(f"  Already processed, loading from {out_path}")
            df = pl.read_parquet(out_path)
        else:
            df = process_movie(
                movie_path,
                model,
                emotions,
                eval_transform,
                face_detector,
                device,
                frame_stride=frame_stride,
                min_face_confidence=min_face_confidence,
            )
            df.write_parquet(out_path)
            print(f"  Saved to {out_path}")

            # Print quick summary
            if len(df) > 0:
                print(f"  Faces detected: {len(df)}")
                print(f"  Avg confidence: {df['confidence'].mean():.3f}")
                emotion_counts = df["emotion"].value_counts().sort("count", descending=True)
                print(f"  Top emotion: {emotion_counts[0, 'emotion']} ({emotion_counts[0, 'count']} faces)")
            else:
                print("  No faces detected.")

        all_results.append(df)

    # Combine all results
    print("\nCombining results...")
    combined_df = pl.concat(all_results)

    # Save combined results
    combined_path = output_dir / "combined_results.parquet"
    combined_df.write_parquet(combined_path)
    print(f"Saved combined results to {combined_path}")

    # Calculate summary statistics
    n_faces = len(combined_df)
    n_movies = len(movie_paths)
    emotion_dist = combined_df["emotion"].value_counts().sort("count", descending=True)

    # Calculate dominant emotion per movie
    movie_dominant = (
        combined_df
        .group_by(["movie", "emotion"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .group_by("movie")
        .head(1)
    )

    dominant_emotion_counts = movie_dominant["emotion"].value_counts().sort("count", descending=True)

    # Print summary
    print(f"\n{'='*60}")
    print("MOVIE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total faces detected: {n_faces}")
    print(f"Movies processed: {n_movies}")
    print(f"Avg faces per movie: {n_faces / n_movies:.1f}")
    print(f"Avg confidence: {combined_df['confidence'].mean():.3f}")

    # Show mean probabilities across all emotions
    prob_cols = [c for c in combined_df.columns if c.startswith("prob_")]
    if prob_cols:
        print(f"\nMean Probabilities:")
        for col in prob_cols:
            emo_name = col.replace("prob_", "")
            print(f"  {emo_name:>10s}: {combined_df[col].mean():.4f}")

    print(f"\nEmotion Distribution (all faces):")
    for row in emotion_dist.iter_rows(named=True):
        pct = row["count"] / n_faces * 100
        print(f"  {row['emotion']:>10s}: {row['count']:5d} ({pct:5.1f}%)")

    print(f"\nDominant Emotion Distribution (per movie):")
    for row in dominant_emotion_counts.iter_rows(named=True):
        pct = row["count"] / n_movies * 100
        print(f"  {row['emotion']:>10s}: {row['count']:3d} movies ({pct:5.1f}%)")
    print(f"{'='*60}\n")

    summary = {
        "total_faces_detected": n_faces,
        "total_movies": n_movies,
        "avg_faces_per_movie": n_faces / n_movies,
        "avg_confidence": combined_df["confidence"].mean(),
        "emotion_distribution": emotion_dist.to_dicts(),
        "dominant_emotion_distribution": dominant_emotion_counts.to_dicts(),
    }

    return summary, combined_df


# -------------------------
# Main Function
# -------------------------

def evaluate_model_on_movies(
    checkpoint_path: Path,
    movies_dir: Path = MOVIES_DIR,
    movie_list_path: Path = MOVIE_LIST,
    output_dir: Optional[Path] = None,
    face_detector_path: Path = FACE_DETECTOR,
    frame_stride: int = 100,
    min_face_confidence: float = 0.5,
    device: str = "cuda",
):
    """
    Evaluate a single model on movies.

    If output_dir is not provided, it is derived as:
        evaluation/<model_name>/movies/
    """
    # Derive model name and output dir
    model_name = derive_model_name(checkpoint_path)
    if output_dir is None:
        output_dir = EVALUATION_ROOT / model_name / "movies"

    # Load movie list
    with open(movie_list_path) as f:
        movie_list = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"MOVIE EVALUATION — {model_name}")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Movies directory: {movies_dir}")
    print(f"Number of movies: {len(movie_list)}")
    print(f"Output directory: {output_dir}")
    print(f"Frame stride: {frame_stride}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model, emotions, eval_transform = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Model loaded. Emotions: {emotions}\n")

    # Build movie paths
    movie_paths = [movies_dir / movie_name for movie_name in movie_list]

    # Check that movies exist
    missing_movies = [p for p in movie_paths if not p.exists()]
    if missing_movies:
        print("Warning: The following movies were not found:")
        for p in missing_movies:
            print(f"  - {p}")
        movie_paths = [p for p in movie_paths if p.exists()]
        print(f"Proceeding with {len(movie_paths)} movies\n")

    if not movie_paths:
        print("Error: No valid movie files found")
        return

    # Evaluate on movies
    summary, movie_df = evaluate_movies(
        movie_paths=movie_paths,
        model=model,
        emotions=emotions,
        eval_transform=eval_transform,
        device=device,
        output_dir=output_dir,
        face_detector_path=face_detector_path,
        frame_stride=frame_stride,
        min_face_confidence=min_face_confidence,
    )


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained model(s) on movies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python src/movie_evaluation/evaluate_movies.py --checkpoint models/pixabay_light_v2/model.pt

  # Multiple models (batch)
  python src/movie_evaluation/evaluate_movies.py --checkpoint models/pixabay_light_v2/model.pt models/pixabay_heavy_v1/model.pt

  # All models under a directory
  python src/movie_evaluation/evaluate_movies.py --checkpoint-dir models/

  # Override defaults
  python src/movie_evaluation/evaluate_movies.py --checkpoint models/pixabay_light_v2/model.pt --frame-stride 50
""",
    )
    parser.add_argument(
        "--checkpoint", nargs="+",
        help="Path(s) to model checkpoint(s)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory to scan for model.pt files (recursive). "
             "Used as batch mode alternative to --checkpoint.",
    )
    parser.add_argument(
        "--movies-dir", default=str(MOVIES_DIR),
        help=f"Directory containing movies (default: {MOVIES_DIR})",
    )
    parser.add_argument(
        "--movie-list", default=str(MOVIE_LIST),
        help=f"Path to file with movie filenames (default: {MOVIE_LIST})",
    )
    parser.add_argument(
        "--face-detector", default=str(FACE_DETECTOR),
        help=f"Path to MediaPipe face detector (default: {FACE_DETECTOR})",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=100,
        help="Process every Nth frame (default: 100)",
    )
    parser.add_argument(
        "--min-face-confidence", type=float, default=0.5,
        help="Minimum face detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to use (default: cuda)",
    )

    args = parser.parse_args()

    # Collect checkpoint paths
    checkpoints = []

    if args.checkpoint:
        checkpoints.extend([Path(p) for p in args.checkpoint])

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        found = sorted(ckpt_dir.rglob("model.pt"))
        print(f"Found {len(found)} checkpoints under {ckpt_dir}:")
        for p in found:
            print(f"  {p}")
        checkpoints.extend(found)

    if not checkpoints:
        parser.error("Provide --checkpoint and/or --checkpoint-dir")

    # Deduplicate (resolve to avoid path aliases)
    seen = set()
    unique_checkpoints = []
    for cp in checkpoints:
        resolved = cp.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_checkpoints.append(cp)

    # Run evaluation for each model
    for i, ckpt in enumerate(unique_checkpoints):
        if len(unique_checkpoints) > 1:
            print(f"\n{'#'*60}")
            print(f"# MODEL {i+1}/{len(unique_checkpoints)}: {derive_model_name(ckpt)}")
            print(f"{'#'*60}")

        evaluate_model_on_movies(
            checkpoint_path=ckpt,
            movies_dir=Path(args.movies_dir),
            movie_list_path=Path(args.movie_list),
            face_detector_path=Path(args.face_detector),
            frame_stride=args.frame_stride,
            min_face_confidence=args.min_face_confidence,
            device=args.device,
        )

    if len(unique_checkpoints) > 1:
        print(f"\nBatch complete. {len(unique_checkpoints)} models evaluated.")
        print(f"Results in: {EVALUATION_ROOT}/")