import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Expressions Ensemble: Emotional Faces in Movies",
    layout="wide",
)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

EMOTION_COLORS = {
    "angry": "#E74C3C",
    "fear": "#7F8C8D",
    "happy": "#FFD93D",
    "sad": "#4A90E2",
    "surprise": "#9B59B6",
}

EMOTIONS = ["angry", "fear", "happy", "sad", "surprise"]

MODEL_LABELS = {
    "ExE": "Expressions Ensemble",
    "FER": "FER-2013",
    "RAF-DB": "RAF-DB",
}

MOVIE_GROUPS = {
    "Featured Selection": [
        "finding_nemo", "airplane", "inside_out", "real_steel",
        "dark_knight", "pulp_fiction", "dodgeball", "inception",
    ],
    "Comedies": [
        "airplane", "dodgeball", "old_school", "pineapple_exp",
        "boondock_saints", "hot_fuzz", "seven_psychopaths", "scott_pilgrim",
    ],
    "Kids Movies": [
        "finding_nemo", "big_hero_6", "inside_out", "lego_movie",
        "frankenweenie",
    ],
    "Harry Potter Series": [
        "hp1_sorcerers_stone", "hp2_chamber_of_secrets",
        "hp3_prisoner_of_azkaban", "hp4_goblet_of_fire",
        "hp5_order_phoenix", "hp6_half_blood_prince",
        "hp7_deathly_hallows_part_1", "hp7_deathly_hallows_part_2",
    ],
    "Dark Dramas": [
        "dark_knight", "dark_knight_rises", "inception",
        "black_mass", "pulp_fiction", "lucky_number_slevin",
    ],
    "Lord of the Rings": ["lotr_1", "lotr_2", "lotr_3"],
}

# Diff chart visual config
DIFF_COLORS = {
    "positive": "#3D7A8A",
    "negative": "#C4737A",
}
BG_COLOR = "#FAFBFC"
GRID_COLOR = "#E2E2E8"
TEXT_COLOR = "#2D3142"
MUTED_TEXT = "#8A8F98"

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_model_data(model_key: str, file_name: str) -> pd.DataFrame:
    """Load a parquet file for a given model."""
    base = Path("movie_data") / model_key
    return pd.read_parquet(base / file_name)


@st.cache_data(show_spinner=False)
def load_all_models():
    """Load all three models' data."""
    models = {}
    for key in ["ExE", "RAFDB", "FER"]:
        models[key] = {
            "timeline": load_model_data(key, "all_movies.parquet"),
            "summary": load_model_data(key, "movie_emotion_summary.parquet"),
            "tidy": load_model_data(key, "movie_emotion_summary_tidy.parquet"),
        }
    return models


data = load_all_models()
ALL_MOVIES = sorted(data["ExE"]["summary"]["movie"].unique())

# Map radio labels to data keys
RADIO_TO_KEY = {
    "Expressions Ensemble": "ExE",
    "FER-2013": "FER",
    "RAF-DB": "RAFDB",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def emotion_legend():
    cols = st.columns(len(EMOTIONS))
    for i, e in enumerate(EMOTIONS):
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background:{EMOTION_COLORS[e]};
                    padding:10px;
                    border-radius:6px;
                    text-align:center;
                    font-weight:600;
                    color:white;
                ">
                    {e.title()}
                </div>
                """,
                unsafe_allow_html=True,
            )


def prepare_stacked_data(df: pd.DataFrame, movies: list) -> pd.DataFrame:
    """Melt wide summary into tidy format for stacked bar chart."""
    df = df[df["movie"].isin(movies)].copy()
    pct_cols = [f"pct_{e}" for e in EMOTIONS]
    tidy = pd.melt(
        df,
        id_vars="movie",
        value_vars=pct_cols,
        var_name="emotion",
        value_name="percent",
    )
    tidy["emotion"] = tidy["emotion"].str.replace("pct_", "")
    tidy.sort_values(by="movie", inplace=True)
    return tidy


def stacked_bar(tidy: pd.DataFrame, title: str):
    fig = px.bar(
        tidy,
        x="movie",
        y="percent",
        color="emotion",
        barmode="stack",
        color_discrete_map=EMOTION_COLORS,
        category_orders={"emotion": EMOTIONS},
        height=550,
        title=title,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(showlegend=False)
    return fig


def prepare_timeline(df: pd.DataFrame, movie: str, threshold: float):
    df = df[(df["movie"] == movie) & (df["confidence"] >= threshold)].copy()
    df.sort_values("timestamp_sec", inplace=True)

    if df.empty:
        return None

    for e in EMOTIONS:
        df[f"{e}_count"] = (df["emotion"] == e).cumsum()

    df["timestamp_min"] = df["timestamp_sec"] / 60
    return df


def timeline_plot(df: pd.DataFrame, movie: str, threshold: float):
    fig = go.Figure()
    for e in EMOTIONS:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_min"],
                y=df[f"{e}_count"],
                mode="lines",
                line=dict(color=EMOTION_COLORS[e], width=2.5),
                name=e,
            )
        )

    fig.update_layout(
        title=f"{movie.replace('_', ' ').title()} (confidence ≥ {threshold})",
        xaxis_title="Time (minutes)",
        yaxis_title="Total faces detected",
        hovermode="x unified",
        height=500,
        showlegend=False,
    )
    return fig


def make_diff_chart(
    tidy_data: dict,
    movies: list,
    model_a: str,
    benchmarks: list[str],
    subset_label: str = "movies",
) -> go.Figure:
    """
    Diverging horizontal bar: model_a minus each benchmark.
    tidy_data: {model_key: DataFrame with movie, emotion, pct columns}
    """
    diffs = {}
    for bench_key in benchmarks:
        bench_label = {"RAFDB": "RAF-DB", "FER": "FER-2013"}.get(bench_key, bench_key)
        a_df = tidy_data[model_a]
        b_df = tidy_data[bench_key]

        a_df = a_df[a_df["movie"].isin(movies)]
        b_df = b_df[b_df["movie"].isin(movies)]

        a_means = a_df.groupby("emotion")["pct"].mean()
        b_means = b_df.groupby("emotion")["pct"].mean()

        diffs[bench_label] = {e: round(a_means.get(e, 0) - b_means.get(e, 0), 1) for e in EMOTIONS}

    n_bench = len(diffs)
    emo_order = list(reversed(EMOTIONS))

    fig = make_subplots(
        rows=1, cols=n_bench,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=["" for _ in diffs],
    )

    for ci, (bench_label, emo_diffs) in enumerate(diffs.items(), 1):
        vals = [emo_diffs[e] for e in emo_order]
        colors = [DIFF_COLORS["positive"] if v >= 0 else DIFF_COLORS["negative"] for v in vals]
        labels = [e.capitalize() for e in emo_order]
        text_vals = [f"+{v:.1f}" if v > 0 else f"{v:.1f}" for v in vals]

        fig.add_trace(
            go.Bar(
                y=labels, x=vals, orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=text_vals,
                textposition=["outside" if abs(v) < 5 else "inside" for v in vals],
                textfont=dict(size=12, color=TEXT_COLOR),
                hovertemplate=f"<b>%{{y}}</b><br>ExE detects %{{x:+.1f}}% more than {bench_label}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=ci,
        )
        fig.add_vline(x=0, row=1, col=ci, line=dict(color=MUTED_TEXT, width=1.5))

    max_abs = max(abs(v) for d in diffs.values() for v in d.values())
    x_range = max(max_abs * 1.4, 5)

    fig.update_layout(
        title=dict(
            text=(
                f"How ExE Differs from Benchmark Models<br>"
                f"<span style='font-size:13px;color:{MUTED_TEXT}'>"
                f"Mean emotion % difference across {len(movies)} {subset_label}</span>"
            ),
            font=dict(size=16, color=TEXT_COLOR),
            x=0.5, xanchor="center", y=0.98, yanchor="top",
        ),
        plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        height=380, margin=dict(l=80, r=30, t=110, b=40),
        bargap=0.3,
    )

    # Directional labels
    bench_list = list(diffs.keys())
    for ci_idx, bench_label in enumerate(bench_list):
        if n_bench == 2:
            x_left  = 0.12 if ci_idx == 0 else 0.57
            x_right = 0.43 if ci_idx == 0 else 0.88
        else:
            x_left, x_right = 0.12, 0.88

        fig.add_annotation(
            text=f"<b>← {bench_label}</b>",
            xref="paper", yref="paper", x=x_left, y=1.0,
            showarrow=False, font=dict(size=12, color=DIFF_COLORS["negative"]),
            xanchor="left", yanchor="bottom",
        )
        fig.add_annotation(
            text="<b>ExE →</b>",
            xref="paper", yref="paper", x=x_right, y=1.0,
            showarrow=False, font=dict(size=12, color=DIFF_COLORS["positive"]),
            xanchor="right", yanchor="bottom",
        )

    for ci in range(1, n_bench + 1):
        fig.update_xaxes(
            range=[-x_range, x_range], showgrid=True,
            gridcolor=GRID_COLOR, gridwidth=0.5, zeroline=False,
            tickfont=dict(size=10, color=MUTED_TEXT), ticksuffix="%",
            row=1, col=ci,
        )
        fig.update_yaxes(tickfont=dict(size=13, color=TEXT_COLOR), showgrid=False, row=1, col=ci)

    # Remove empty subplot annotations
    fig.layout.annotations = [a for a in fig.layout.annotations if a.text != ""]

    return fig


def get_model_keys(selection: str) -> list[str]:
    """Convert radio selection to list of data keys."""
    if selection == "Side-by-side":
        return ["ExE", "RAFDB"]
    elif selection == "All three":
        return ["ExE", "RAFDB", "FER"]
    else:
        return [RADIO_TO_KEY[selection]]


def display_label(key: str) -> str:
    return {"ExE": "Expressions Ensemble", "RAFDB": "RAF-DB", "FER": "FER-2013"}.get(key, key)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state():
    defaults = {
        "compare_group": "Comedies",
        "compare_movies": MOVIE_GROUPS["Comedies"][:6],
        "compare_model": "Side-by-side",
        "timeline_movie": ALL_MOVIES[0],
        "timeline_model": "Expressions Ensemble",
        "timeline_conf": 0.5,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()

# =============================================================================
# HEADER
# =============================================================================

st.title("Expressions Ensemble: Emotional Faces in Movies")
st.markdown(
    """
    This project explores how different emotion recognition models behave
    when applied to real movies.

    Standard benchmark datasets often feature overly exaggerated poses with
    limited ecological validity. Here, rather than relying on test accuracy
    alone, we ask:

    **Do model predictions make sense in context?**
    """
)

emotion_legend()
st.markdown("---")

# =============================================================================
# TABS
# =============================================================================

tab_overview, tab_explore, tab_timeline, tab_methods = st.tabs(
    [
        "🎬 Overview",
        "📊 Movie Explorer",
        "📈 Timeline Stories",
        "🧭 Methods & Insight",
    ]
)

# =============================================================================
# TAB 1 — OVERVIEW
# =============================================================================

with tab_overview:
    st.subheader("What does emotion look like across movies?")

    st.markdown(
        """
        The chart below shows how ExE's emotion predictions differ from two
        widely-used benchmark models across all 59 films in the dataset.

        Bars extending **right** mean ExE detects **more** of that emotion;
        bars extending **left** mean the benchmark detects more.
        """
    )

    # Diff chart across all movies
    tidy_all = {k: v["tidy"] for k, v in data.items()}
    fig_diff = make_diff_chart(
        tidy_all, ALL_MOVIES,
        model_a="ExE",
        benchmarks=["FER", "RAFDB"],
        subset_label="movies",
    )
    st.plotly_chart(fig_diff, use_container_width=True, key="overview_diff")

    st.markdown(
        """
        **What to notice:**
        - FER-2013 heavily over-predicts **angry** and **fear** at the expense
          of every other emotion — even in comedies and kids' movies
        - RAF-DB tilts toward **surprise**, underdetecting fear relative to ExE
        - ExE finds a broader, more genre-appropriate mix of emotions
        """
    )

    st.markdown("---")

    st.markdown("#### Comedy close-up")
    st.markdown(
        """
        The pattern is easiest to see in comedies, where we'd expect
        meaningful **happy** and **surprise** alongside lighter emotions.
        """
    )

    # Comedy stacked bars — side by side for ExE and RAF-DB
    comedy_movies = MOVIE_GROUPS["Comedies"]
    col_exe, col_raf = st.columns(2)
    with col_exe:
        st.plotly_chart(
            stacked_bar(
                prepare_stacked_data(data["ExE"]["summary"], comedy_movies),
                "Expressions Ensemble",
            ),
            use_container_width=True,
            key="overview_comedy_exe",
        )
    with col_raf:
        st.plotly_chart(
            stacked_bar(
                prepare_stacked_data(data["RAFDB"]["summary"], comedy_movies),
                "RAF-DB",
            ),
            use_container_width=True,
            key="overview_comedy_raf",
        )


# =============================================================================
# TAB 2 — MOVIE EXPLORER
# =============================================================================

with tab_explore:
    # Callback: when group changes, update the movie multiselect
    def on_group_change():
        group = st.session_state["explore_group"]
        st.session_state["explore_movies"] = MOVIE_GROUPS[group]

    # Initialize multiselect state if not yet set
    if "explore_movies" not in st.session_state:
        st.session_state["explore_movies"] = MOVIE_GROUPS["Comedies"]

    with st.expander("⚙️ Explore settings", expanded=True):
        c1, c2, c3 = st.columns([2, 1.5, 3])

        with c1:
            model_view = st.radio(
                "View",
                ["Expressions Ensemble", "RAF-DB", "FER-2013", "Side-by-side", "All three"],
                horizontal=True,
                key="explore_model_radio",
            )

        with c2:
            st.selectbox(
                "Movie group",
                list(MOVIE_GROUPS.keys()),
                key="explore_group",
                on_change=on_group_change,
            )

        with c3:
            movies = st.multiselect(
                "Movies",
                ALL_MOVIES,
                max_selections=10,
                format_func=lambda x: x.replace("_", " ").title(),
                key="explore_movies",
            )

    st.markdown("---")

    if not movies:
        st.warning("Select at least one movie to continue.")
        st.stop()

    model_keys = get_model_keys(model_view)

    if len(model_keys) == 1:
        mk = model_keys[0]
        st.plotly_chart(
            stacked_bar(
                prepare_stacked_data(data[mk]["summary"], movies),
                display_label(mk),
            ),
            use_container_width=True,
            key=f"explore_stack_{mk}",
        )
    elif len(model_keys) == 2:
        cols = st.columns(2)
        for col, mk in zip(cols, model_keys):
            with col:
                st.plotly_chart(
                    stacked_bar(
                        prepare_stacked_data(data[mk]["summary"], movies),
                        display_label(mk),
                    ),
                    use_container_width=True,
                    key=f"explore_stack_{mk}",
                )
    else:
        # All three — use tabs instead of cramped columns
        m_tabs = st.tabs([display_label(k) for k in model_keys])
        for m_tab, mk in zip(m_tabs, model_keys):
            with m_tab:
                st.plotly_chart(
                    stacked_bar(
                        prepare_stacked_data(data[mk]["summary"], movies),
                        display_label(mk),
                    ),
                    use_container_width=True,
                    key=f"explore_stack_{mk}",
                )

    # Diff chart for selected movies
    if len(movies) >= 2:
        st.markdown("---")
        st.markdown("#### Difference from benchmarks (selected movies)")
        tidy_all = {k: v["tidy"] for k, v in data.items()}
        fig = make_diff_chart(
            tidy_all, movies,
            model_a="ExE",
            benchmarks=["FER", "RAFDB"],
            subset_label="selected movies",
        )
        st.plotly_chart(fig, use_container_width=True, key="explore_diff")


# =============================================================================
# TAB 3 — TIMELINE STORIES
# =============================================================================

with tab_timeline:
    with st.expander("⚙️ Timeline settings", expanded=True):
        c1, c2, c3 = st.columns([2, 2.5, 2])

        with c1:
            tl_model = st.radio(
                "View",
                ["Expressions Ensemble", "RAF-DB", "FER-2013", "Side-by-side"],
                horizontal=True,
                key="timeline_model_radio",
            )

        with c2:
            tl_movie = st.selectbox(
                "Movie",
                ALL_MOVIES,
                format_func=lambda x: x.replace("_", " ").title(),
                key="timeline_movie_select",
            )

        with c3:
            tl_conf = st.slider(
                "Minimum confidence",
                0.0, 1.0, 0.5, 0.05,
                key="timeline_conf_slider",
            )

    st.markdown("---")

    def show_timeline(model_key: str):
        df = data[model_key]["timeline"]
        prepped = prepare_timeline(df, tl_movie, tl_conf)
        if prepped is None:
            st.warning("No data available at this confidence level.")
        else:
            st.plotly_chart(
                timeline_plot(prepped, tl_movie, tl_conf),
                use_container_width=True,
                key=f"timeline_{model_key}",
            )

    tl_keys = get_model_keys(tl_model)

    if len(tl_keys) == 1:
        show_timeline(tl_keys[0])
    else:
        cols = st.columns(len(tl_keys))
        for col, key in zip(cols, tl_keys):
            with col:
                st.markdown(f"**{display_label(key)}**")
                show_timeline(key)


# =============================================================================
# TAB 4 — METHODS & INSIGHT (MERGED)
# =============================================================================

with tab_methods:
    st.subheader("How this works (and where it breaks)")

    st.markdown(
        """
        #### Approach

        This project uses **weak supervision**: training images are collected
        from stock photo sites using emotion-related search keywords rather
        than relying on hand-labeled benchmark sets.

        The **Expressions Ensemble (ExE)** averages predictions from four
        models — two trained on Pexels images, two on Pixabay — to produce
        a more stable emotion signal. These are compared against two standard
        benchmarks: **FER-2013** and **RAF-DB**.

        All models evaluate the same movie frames: faces detected every 100th
        frame across 59 films spanning comedy, drama, action, horror, and
        animation.
        """
    )

    st.markdown(
        """
        #### Limitations

        - **No ground truth** for movie emotions — evaluation is based on
          whether patterns match genre expectations, not frame-level labels
        - Some emotions are over- or under-represented depending on the
          training source
        - Timeline interpretation is qualitative; cumulative counts show
          trends, not precise moments
        - Models are research demonstrations, not production classifiers
        - The 59-film dataset skews toward English-language action and
          genre films

        The goal is not perfect accuracy, but **meaningful patterns in
        realistic settings**.
        """
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built by <a href='https://pixelprocess.org' target='_blank'>PixelProcess</a> |
        Part of <a href='https://dexterousdata.com' target='_blank'>Dexterous Data</a><br>
        <a href='https://github.com/pixel-process-dev/expressions-ensemble' target='_blank'>
        View Source on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)