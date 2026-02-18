"""
ExE Difference Score Chart — Comedy Emotion Distributions
==========================================================
Diverging horizontal bar chart showing (ExE % - Benchmark %)
for each emotion across comedy films.

Reusable: swap in different movie subsets or model pairs.

Requirements: plotly, polars (or swap to pandas)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these to reuse with other subsets
# ---------------------------------------------------------------------------
EVAL_ROOT = Path("evaluation")

MODELS = {
    "ExE":    EVAL_ROOT / "ensemble",
    "FER":    EVAL_ROOT / "fer_2013",
    "RAF-DB": EVAL_ROOT / "raf_db",
}

# Benchmarks to compare against ExE
BENCHMARKS = ["FER", "RAF-DB"]

# Movie subset to include (None = all movies)
MOVIE_SUBSET = [
    '2_guns',
    '300',
    '47_ronin',
    '9',
    'airplane',
    'amazing_spider-man_the',
    'avengers_age_of_ultron',
    'big_hero_6',
    'black_mass',
    'blackkklansman',
    'boondock_saints',
    'chronicles_of_riddick',
    'dark_knight',
    'dark_knight_rises',
    'dark_shadows',
    'dazed_and_confused',
    'diehard',
    'dodgeball',
    'domino',
    'finding_nemo',
    'frankenweenie',
    'gaffigan_kingbaby',
    'hitchhikersguidetothegalaxy',
    'hot_fuzz',
    'hp1_sorcerers_stone',
    'hp2_chamber_of_secrets',
    'hp3_prisoner_of_azkaban',
    'hp4_goblet_of_fire',
    'hp5_order_phoenix',
    'hp6_half_blood_prince',
    'hp7_deathly_hallows_part_1',
    'hp7_deathly_hallows_part_2',
    'i_am_legend',
    'inception',
    'inside_out',
    'iron_man',
    'iron_man_2',
    'iron_man_3',
    'jurassic_world',
    'kick_ass',
    'kingsman',
    'lego_movie',
    'lotr_1',
    'lotr_2',
    'lotr_3',
    'lucky_number_slevin',
    'mad_max_fury_road',
    'mad_max_thunderdome',
    'old_school',
    'once_upon_a_time_in_mexico_t00',
    'pacific_rim',
    'pineapple_exp',
    'pitch_black',
    'point_break',
    'pulp_fiction',
    'real_steel',
    'scott_pilgrim',
    'serenity',
    'seven_psychopaths'
]




SUBSET_LABEL = "movies"  # used in title

EMOTIONS = ["happy", "surprise", "sad", "fear", "angry"]

# Visual config
COLORS = {
    "positive": "#3D7A8A",  # ExE navy-teal (ExE detects more)
    "negative": "#C4737A",  # muted rose (benchmark detects more)
}
BG_COLOR = "#FAFBFC"
GRID_COLOR = "#E2E2E8"
TEXT_COLOR = "#2D3142"
MUTED_TEXT = "#8A8F98"


# ---------------------------------------------------------------------------
# LOAD & COMPUTE
# ---------------------------------------------------------------------------
def load_tidy_summaries() -> pl.DataFrame:
    """Load tidy emotion summaries from all models."""
    frames = []
    for label, path in MODELS.items():
        fp = path / "movie_emotion_summary_tidy.parquet"
        print(fp)
        if not fp.exists():
            print(f"  ⚠ Missing: {fp}")
            continue
        df = pl.read_parquet(fp).with_columns(pl.lit(label).alias("model"))
        frames.append(df)
    return pl.concat(frames)


def compute_mean_diffs(data: pl.DataFrame, movie_subset: list[str] | None = None) -> dict:
    """
    Compute mean emotion % per model, then return ExE - benchmark diffs.

    Returns:
        {benchmark_name: {emotion: diff_value, ...}, ...}
    """
    if movie_subset:
        data = data.filter(pl.col("movie").is_in(movie_subset))

    # Mean pct per model × emotion
    means = (
        data.group_by(["model", "emotion"])
        .agg(pl.col("pct").mean().alias("mean_pct"))
    )

    exe_means = {
        row["emotion"]: row["mean_pct"]
        for row in means.filter(pl.col("model") == "ExE").iter_rows(named=True)
    }

    diffs = {}
    for bench in BENCHMARKS:
        bench_means = {
            row["emotion"]: row["mean_pct"]
            for row in means.filter(pl.col("model") == bench).iter_rows(named=True)
        }
        diffs[bench] = {
            emo: round(exe_means.get(emo, 0) - bench_means.get(emo, 0), 2)
            for emo in EMOTIONS
        }

    return diffs, exe_means


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
def make_diff_chart(
    diffs: dict,
    exe_means: dict,
    subset_label: str = "All Movies",
    show: bool = True,
    save_path: str | None = None,
) -> go.Figure:
    """
    Create diverging horizontal bar chart.

    Args:
        diffs: {benchmark: {emotion: diff, ...}}
        exe_means: {emotion: mean_pct} for annotation context
        subset_label: label for the movie subset
        show: whether to display in browser
        save_path: optional path to save as HTML or image
    """
    n_benchmarks = len(diffs)

    fig = make_subplots(
        rows=1, cols=n_benchmarks,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=["" for _ in diffs.keys()],
    )

    # Emotion display order (top to bottom on horizontal bars)
    emo_order = list(reversed(EMOTIONS))

    for ci, (bench, emo_diffs) in enumerate(diffs.items(), 1):
        vals = [emo_diffs[e] for e in emo_order]
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals]
        labels = [e.capitalize() for e in emo_order]

        # Text annotations showing the diff value
        text_vals = [f"+{v:.1f}" if v > 0 else f"{v:.1f}" for v in vals]

        fig.add_trace(
            go.Bar(
                y=labels,
                x=vals,
                orientation="h",
                marker=dict(
                    color=colors,
                    cornerradius=4,
                    line=dict(width=0),
                ),
                text=text_vals,
                textposition=["outside" if abs(v) < 5 else "inside" for v in vals],
                textfont=dict(size=12, color=TEXT_COLOR, family="DM Sans"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "ExE detects %{x:+.1f}% more than " + bench + "<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=ci,
        )

        # Zero line
        fig.add_vline(
            x=0, row=1, col=ci,
            line=dict(color=MUTED_TEXT, width=1.5, dash="solid"),
        )

    # Layout
    max_abs = max(
        abs(v)
        for bench_diffs in diffs.values()
        for v in bench_diffs.values()
    )
    x_range = max_abs * 1.4  # pad for text

    fig.update_layout(
        title=dict(
            text=(
                f"<br>How ExE Differs from Benchmark Models<br>"
                f"<span style='font-size:13px;color:{MUTED_TEXT}'>"
                f"Mean emotion % difference across {len(MOVIE_SUBSET) if MOVIE_SUBSET else 'all'} "
                f"{subset_label.lower()}</span>"
            ),
            font=dict(size=18, color=TEXT_COLOR, family="DM Sans"),
            x=0.5,
            xanchor="center",
            y=0.96,
            yanchor="top",
        ),
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(family="DM Sans", color=TEXT_COLOR),
        height=380,
        width=750,
        margin=dict(l=90, r=40, t=115, b=40),
        bargap=0.3,
    )

    # Add directional labels: benchmark name on left, ExE on right
    bench_list = list(diffs.keys())
    for ci_idx, bench in enumerate(bench_list):
        if n_benchmarks == 2:
            # Left panel x-domain ~0.0-0.46, right panel ~0.54-1.0
            x_left  = 0.12 if ci_idx == 0 else 0.57
            x_right = 0.43 if ci_idx == 0 else 0.88
        else:
            x_left, x_right = 0.12, 0.88

        fig.add_annotation(
            text=f"<b>\u2190 {bench}</b>",
            xref="paper", yref="paper",
            x=x_left, y=1.0,
            showarrow=False,
            font=dict(size=12, color=COLORS["negative"], family="DM Sans"),
            xanchor="left", yanchor="bottom",
        )
        fig.add_annotation(
            text=f"<b>ExE \u2192</b>",
            xref="paper", yref="paper",
            x=x_right, y=1.0,
            showarrow=False,
            font=dict(size=12, color=COLORS["positive"], family="DM Sans"),
            xanchor="right", yanchor="bottom",
        )

    # Style axes
    for ci in range(1, n_benchmarks + 1):
        fig.update_xaxes(
            range=[-x_range, x_range],
            showgrid=True,
            gridcolor=GRID_COLOR,
            gridwidth=0.5,
            zeroline=False,
            tickfont=dict(size=10, color=MUTED_TEXT),
            ticksuffix="%",
            row=1, col=ci,
        )
        fig.update_yaxes(
            tickfont=dict(size=13, color=TEXT_COLOR),
            showgrid=False,
            row=1, col=ci,
        )

    # Clean up empty subplot title annotations from make_subplots
    fig.layout.annotations = [a for a in fig.layout.annotations if a.text != ""]

    if save_path:
        p = Path(save_path)
        if p.suffix == ".html":
            fig.write_html(str(p), include_plotlyjs="cdn")
        else:
            fig.write_image(str(p), scale=2)
        print(f"✓ Saved: {p}")

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    data = load_tidy_summaries()
    diffs, exe_means = compute_mean_diffs(data, MOVIE_SUBSET)

    print("\nDifference scores (ExE - benchmark):")
    print("-" * 50)
    for bench, emo_diffs in diffs.items():
        print(f"\n  ExE vs {bench}:")
        for emo in EMOTIONS:
            d = emo_diffs[emo]
            direction = "ExE +" if d > 0 else f"{bench} +"
            print(f"    {emo:<10}  {d:>+6.1f}%  ({direction})")

    # Generate figure
    fig = make_diff_chart(
        diffs,
        exe_means,
        subset_label=SUBSET_LABEL,
        show=True,
        save_path=str(EVAL_ROOT / "ensemble" / "all_diff_chart.png"),
    )

    # Also save interactive HTML
    fig.write_html(
        str(EVAL_ROOT / "ensemble" / "all_diff_chart.html"),
        include_plotlyjs="cdn",
    )
    print(f"✓ Saved interactive: {EVAL_ROOT / 'ensemble' / 'all_movies_diff_chart.html'}")


if __name__ == "__main__":
    main()