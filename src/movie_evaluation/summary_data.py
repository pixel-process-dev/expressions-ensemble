"""
Expressions Ensemble (ExE) - Movie Analysis Pipeline
=====================================================
Builds per-model and ensemble summary files from individual movie parquet files.

Outputs per model directory:
  - all_movies.parquet           : concatenated frames from all 59 movies
  - movie_emotion_summary.parquet: 1 row per movie (wide)
  - movie_emotion_summary_tidy.parquet: 1 row per movie×emotion (long)

Ensemble outputs (from pexels + pixabay models only):
  - all_movies.parquet
  - movie_emotion_summary.parquet
  - movie_emotion_summary_tidy.parquet
"""

import polars as pl
from pathlib import Path
import warnings

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EVAL_ROOT = Path("evaluation/")

ALL_MODELS = [
    "pexels_light_aug_v2",
    "pexels_light_aug_v1",
    "fer_2013",
    "pixabay_light_aug_v1",
    "raf_db",
    "pixabay_light_aug_v2",
]

ENSEMBLE_MODELS = [
    "pexels_light_aug_v2",
    "pexels_light_aug_v1",
    "pixabay_light_aug_v1",
    "pixabay_light_aug_v2",
]

EMOTIONS = ["angry", "fear", "happy", "sad", "surprise"]
PROB_COLS = [f"prob_{e}" for e in EMOTIONS]

ENSEMBLE_DIR = EVAL_ROOT / "ensemble"

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_all_movies(model_dir: Path) -> pl.DataFrame:
    """Load and concatenate all movie parquet files for a model."""
    movies_dir = model_dir / "movies"
    parquet_files = sorted(movies_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {movies_dir}")

    print(f"  Loading {len(parquet_files)} movie files from {movies_dir}")

    frames = []
    for f in parquet_files:
        df = pl.read_parquet(f)
        # Ensure movie col exists (derive from filename if missing)
        if "movie" not in df.columns:
            movie_name = f.stem
            df = df.with_columns(pl.lit(movie_name).alias("movie"))
        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed")

    # Add model label
    model_name = model_dir.name
    if "model" not in combined.columns:
        combined = combined.with_columns(pl.lit(model_name).alias("model"))

    # Drop unnamed/auto-index columns if present
    drop_cols = [
        c for c in combined.columns
        if c.startswith("__index")
        or c == ""
        or c.startswith("Unnamed")
        or (c.isdigit() and c not in ["movie", "frame_idx"])
    ]
    if drop_cols:
        combined = combined.drop(drop_cols)

    return combined


def build_movie_summary(all_movies: pl.DataFrame) -> pl.DataFrame:
    """
    Build wide-format summary: 1 row per movie.
    Includes total_frames, per-emotion counts, percentages, and mean probabilities.
    """
    model_name = all_movies["model"][0]

    # Total frames per movie
    totals = all_movies.group_by("movie").agg(
        pl.len().alias("total_frames")
    )

    # Per-emotion counts
    emotion_counts = (
        all_movies.group_by(["movie", "emotion"])
        .agg(pl.len().alias("count"))
        .pivot(on="emotion", index="movie", values="count")
        .fill_null(0)
    )
    # Rename pivoted columns to count_<emotion>
    rename_map = {e: f"count_{e}" for e in EMOTIONS if e in emotion_counts.columns}
    emotion_counts = emotion_counts.rename(rename_map)
    # Ensure all count columns exist
    for e in EMOTIONS:
        col_name = f"count_{e}"
        if col_name not in emotion_counts.columns:
            emotion_counts = emotion_counts.with_columns(pl.lit(0).alias(col_name))

    # Mean probabilities per movie
    mean_probs = all_movies.group_by("movie").agg(
        [pl.col(pc).mean().alias(f"mean_{pc}") for pc in PROB_COLS]
    )

    # Join everything
    summary = totals.join(emotion_counts, on="movie").join(mean_probs, on="movie")

    # Add percentage columns
    for e in EMOTIONS:
        summary = summary.with_columns(
            (pl.col(f"count_{e}") / pl.col("total_frames") * 100)
            .round(2)
            .alias(f"pct_{e}")
        )

    # Add model column
    summary = summary.with_columns(pl.lit(model_name).alias("model"))

    # Sort columns for readability
    id_cols = ["movie", "model", "total_frames"]
    count_cols = sorted([c for c in summary.columns if c.startswith("count_")])
    pct_cols = sorted([c for c in summary.columns if c.startswith("pct_")])
    prob_cols = sorted([c for c in summary.columns if c.startswith("mean_prob_")])
    col_order = id_cols + count_cols + pct_cols + prob_cols
    summary = summary.select(col_order)

    return summary.sort("movie")


def make_tidy(summary: pl.DataFrame) -> pl.DataFrame:
    """
    Convert wide summary to tidy (long) format: 1 row per movie × emotion.
    Columns: movie, model, total_frames, emotion, count, pct, mean_prob
    """
    rows = []
    for e in EMOTIONS:
        subset = summary.select([
            "movie", "model", "total_frames",
            pl.lit(e).alias("emotion"),
            pl.col(f"count_{e}").alias("count"),
            pl.col(f"pct_{e}").alias("pct"),
            pl.col(f"mean_prob_{e}").alias("mean_prob"),
        ])
        rows.append(subset)

    tidy = pl.concat(rows).sort(["movie", "emotion"])
    return tidy


def process_model(model_name: str) -> pl.DataFrame:
    """Process a single model: load, summarize, save. Returns all_movies df."""
    model_dir = EVAL_ROOT / model_name
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    # 1. Build all_movies
    all_movies = load_all_movies(model_dir)
    out_path = model_dir / "all_movies.parquet"
    all_movies.write_parquet(out_path)
    print(f"  ✓ Saved {out_path}  ({all_movies.shape[0]:,} rows × {all_movies.shape[1]} cols)")

    # 2. Build wide summary
    summary = build_movie_summary(all_movies)
    out_path = model_dir / "movie_emotion_summary.parquet"
    summary.write_parquet(out_path)
    print(f"  ✓ Saved {out_path}  ({summary.shape[0]} movies)")

    # 3. Build tidy summary
    tidy = make_tidy(summary)
    out_path = model_dir / "movie_emotion_summary_tidy.parquet"
    tidy.write_parquet(out_path)
    print(f"  ✓ Saved {out_path}  ({tidy.shape[0]} rows)")

    return all_movies


def build_ensemble(ensemble_frames: list[pl.DataFrame]):
    """
    Build ensemble predictions from the 4 custom models.

    Strategy: For each frame (movie × frame_idx), average the probability
    columns across models, then assign emotion = argmax of averaged probs.
    """
    print(f"\n{'='*60}")
    print(f"Building ENSEMBLE from {len(ensemble_frames)} models")
    print(f"{'='*60}")

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    # Combine all ensemble model frames
    combined = pl.concat(ensemble_frames, how="diagonal_relaxed")

    # Key columns for grouping: movie + frame_idx identifies a unique observation
    group_keys = ["movie", "frame_idx"]

    # Columns to average (probabilities)
    # Also keep metadata from first occurrence
    meta_cols = ["timestamp_sec", "face_area", "relative_face_area"]

    # Average probabilities and take first of metadata
    agg_exprs = [pl.col(pc).mean().alias(pc) for pc in PROB_COLS]
    agg_exprs += [pl.col(mc).first().alias(mc) for mc in meta_cols if mc in combined.columns]
    agg_exprs += [pl.len().alias("n_models")]  # track how many models contributed

    ensemble = combined.group_by(group_keys).agg(agg_exprs)

    # Assign ensemble emotion = argmax of averaged probs
    ensemble = ensemble.with_columns(
        pl.concat_list(PROB_COLS)
        .list.arg_max()
        .alias("_argmax")
    )

    emotion_map = {i: e for i, e in enumerate(EMOTIONS)}
    ensemble = ensemble.with_columns(
        pl.col("_argmax")
        .replace_strict(emotion_map)
        .alias("emotion")
    ).drop("_argmax")

    # Confidence = max probability (the winning emotion's averaged prob)
    ensemble = ensemble.with_columns(
        pl.max_horizontal(PROB_COLS).alias("confidence")
    )

    # Add model label
    ensemble = ensemble.with_columns(pl.lit("ensemble").alias("model"))

    # Sort for consistency
    ensemble = ensemble.sort(["movie", "frame_idx"])

    # Save all_movies
    out_path = ENSEMBLE_DIR / "all_movies.parquet"
    ensemble.write_parquet(out_path)
    print(f"  ✓ Saved {out_path}  ({ensemble.shape[0]:,} rows × {ensemble.shape[1]} cols)")

    # Build and save summary
    summary = build_movie_summary(ensemble)
    out_path = ENSEMBLE_DIR / "movie_emotion_summary.parquet"
    summary.write_parquet(out_path)
    print(f"  ✓ Saved {out_path}  ({summary.shape[0]} movies)")

    # Build and save tidy
    tidy = make_tidy(summary)
    out_path = ENSEMBLE_DIR / "movie_emotion_summary_tidy.parquet"
    tidy.write_parquet(out_path)
    print(f"  ✓ Saved {out_path}  ({tidy.shape[0]} rows)")

    return ensemble


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Expressions Ensemble (ExE) - Movie Analysis Pipeline")
    print("=" * 60)

    # Process all individual models
    all_model_frames = {}
    for model_name in ALL_MODELS:
        model_dir = EVAL_ROOT / model_name
        if not (model_dir / "movies").exists():
            print(f"  ⚠ Skipping {model_name}: no movies/ directory found")
            continue
        all_model_frames[model_name] = process_model(model_name)

    # Build ensemble from custom models only
    ensemble_frames = [
        all_model_frames[m] for m in ENSEMBLE_MODELS
        if m in all_model_frames
    ]

    if len(ensemble_frames) == len(ENSEMBLE_MODELS):
        build_ensemble(ensemble_frames)
    else:
        found = [m for m in ENSEMBLE_MODELS if m in all_model_frames]
        missing = [m for m in ENSEMBLE_MODELS if m not in all_model_frames]
        print(f"\n⚠ Ensemble incomplete: found {found}, missing {missing}")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()