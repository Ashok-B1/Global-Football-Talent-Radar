import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Global Football Talent Radar",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent

RADAR_FILE = BASE / "final_talent_radar.csv"
CURRENT_FILE = BASE / "current_talent_radar.csv"
BACKTEST_FILE = BASE / "walk_forward_results.csv"
SUMMARY_FILE = BASE / "model_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    radar = pd.read_csv(RADAR_FILE)
    current = pd.read_csv(CURRENT_FILE)
    backtest = pd.read_csv(BACKTEST_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    return radar, current, backtest, summary


radar, current, backtest, summary = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("⚽ Global Football Talent Radar")

st.caption(
    "ML-powered scouting system for identifying emerging players "
    "in feeder leagues before major-market transfers."
)

st.caption("Data through June 2026 | 6 emerging European leagues")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Scouting Filters")

leagues = sorted(
    radar["competition_name"].dropna().unique()
)

positions = sorted(
    radar["position"].dropna().unique()
)

selected_leagues = st.sidebar.multiselect(
    "League",
    leagues,
    default=leagues
)

selected_positions = st.sidebar.multiselect(
    "Position",
    positions,
    default=positions
)

max_age = st.sidebar.slider(
    "Maximum Age",
    min_value=int(radar["age_at_season_end"].min()),
    max_value=int(radar["age_at_season_end"].max()),
    value=23
)

max_value = st.sidebar.slider(
    "Market Value Ceiling ( M)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.5
)

min_probability = st.sidebar.slider(
    "Minimum Breakout Probability (%)",
    min_value=0,
    max_value=100,
    value=70
)


# ============================================================
# FILTER
# ============================================================

filtered = radar[
    radar["competition_name"].isin(selected_leagues)
    & radar["position"].isin(selected_positions)
    & (radar["age_at_season_end"] <= max_age)
    & (radar["market_value_m"] <= max_value)
    & (
        radar["breakout_probability_pct"]
        >= min_probability
    )
].copy()


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Players Scored",
    f"{len(current):,}"
)

c2.metric(
    "Scouting Candidates",
    f"{len(filtered):,}"
)

c3.metric(
    "Leagues Covered",
    f"{len(radar['competition_name'].unique()):,}"
)

best_probability = (
    filtered["breakout_probability_pct"].max()
    if len(filtered)
    else 0
)

c4.metric(
    "Highest Breakout Score",
    f"{best_probability:.1f}%"
)


st.divider()


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Talent Radar",
        "👤 Player Profile",
        "⚖️ Compare Players",
        "📊 Historical Backtest",
        "🧠 Model Insights"
    ]
)


# ============================================================
# TAB 1 — TALENT RADAR
# ============================================================

with tab1:

    st.subheader("Emerging Talent Radar")

    st.write(
        "Players meeting the selected age, market-value, "
        "playing-time and model-score criteria."
    )

    display_columns = [
        "scouting_rank",
        "player_name",
        "club_name",
        "competition_name",
        "age_at_season_end",
        "position",
        "minutes",
        "goals",
        "assists",
        "market_value_m",
        "breakout_probability_pct"
    ]

    table = filtered[
        display_columns
    ].head(50).copy()

    table.columns = [
        "Rank",
        "Player",
        "Club",
        "League",
        "Age",
        "Position",
        "Minutes",
        "Goals",
        "Assists",
        "Market Value (€M)",
        "Model-Estimated Breakout Probability (%)"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Market Value vs Breakout Probability")

    if len(filtered) > 0:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            filtered["market_value_m"],
            filtered["breakout_probability_pct"],
            alpha=0.7
        )

        ax.set_xlabel(
            "Market Value (€M)"
        )

        ax.set_ylabel(
            "Breakout Probability (%)"
        )

        ax.set_title(
            "Emerging Players: Market Value vs Model Probability"
        )

        ax.grid(alpha=0.2)

        st.pyplot(
            fig,
            use_container_width=True
        )

    st.subheader("Candidates by League")

    league_counts = (
        filtered["competition_name"]
        .value_counts()
    )

    st.bar_chart(
        league_counts
    )


# ============================================================
# TAB 2 — PLAYER PROFILE
# ============================================================

with tab2:

    st.subheader("Player Profile")

    players = sorted(
        filtered["player_name"]
        .dropna()
        .unique()
    )

    if len(players) == 0:

        st.warning(
            "No players match the selected filters."
        )

    else:

        selected_player = st.selectbox(
            "Select Player",
            players
        )

        p = filtered[
            filtered["player_name"]
            == selected_player
        ].iloc[0]

        st.markdown(
            f"## {p['player_name']}"
        )

        st.caption(
            f"{p['club_name']} · "
            f"{p['competition_name']}"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Age",
            f"{int(p['age_at_season_end'])}"
        )

        b.metric(
            "Market Value",
            f"€{p['market_value_m']:.1f}M"
        )

        c.metric(
            "Model-Estimated Breakout",
            f"{p['breakout_probability_pct']:.1f}%"
        )

        d.metric(
            "Minutes",
            f"{int(p['minutes']):,}"
        )

        st.divider()

        x, y, z = st.columns(3)

        x.metric(
            "Goals",
            f"{int(p['goals'])}"
        )

        y.metric(
            "Assists",
            f"{int(p['assists'])}"
        )

        z.metric(
            "Position",
            str(p["position"])
        )

        st.subheader("Performance Profile")

        metrics = pd.DataFrame({
            "Metric": [
                "Goals / 90",
                "Assists / 90",
                "Minutes / 1000"
            ],
            "Value": [
                p["goals_per90"],
                p["assists_per90"],
                p["minutes"] / 1000
            ]
        })

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The breakout probability is a model-estimated "
            "probability of meeting the project's historical "
            "breakout criterion. It is not a guarantee of future success."
        )


# ============================================================
# TAB 3 — PLAYER COMPARISON
# ============================================================

with tab3:

    st.subheader("Player Comparison")

    candidates = sorted(
        filtered["player_name"]
        .dropna()
        .unique()
    )

    if len(candidates) >= 2:

        selected = st.multiselect(
            "Select 2–4 players",
            candidates,
            default=candidates[:2],
            max_selections=4
        )

        if selected:

            comp = filtered[
                filtered["player_name"].isin(selected)
            ].copy()

            comparison_columns = [
                "player_name",
                "age_at_season_end",
                "position",
                "minutes",
                "goals",
                "assists",
                "goals_per90",
                "assists_per90",
                "market_value_m",
                "breakout_probability_pct"
            ]

            comp = comp[
                comparison_columns
            ]

            comp.columns = [
                "Player",
                "Age",
                "Position",
                "Minutes",
                "Goals",
                "Assists",
                "Goals / 90",
                "Assists / 90",
                "Market Value (€M)",
                "Breakout Probability (%)"
            ]

            st.dataframe(
                comp,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "Select broader filters to compare players."
        )


# ============================================================
# TAB 4 — HISTORICAL BACKTEST
# ============================================================

with tab4:

    st.subheader("Walk-Forward Backtest")

    st.write(
        "Each test season is evaluated using only "
        "information available before that season."
    )

    summary_display = summary.copy()

    st.dataframe(
        summary_display.round(3),
        use_container_width=True
    )

    st.subheader(
        "Precision@20 by Test Season"
    )

    if "precision_at_20" in backtest.columns:

        pivot = (
            backtest.pivot(
                index="test_season",
                columns="model",
                values="precision_at_20"
            )
        )

        st.line_chart(
            pivot
        )

    st.subheader(
        "ROC-AUC by Test Season"
    )

    if "roc_auc" in backtest.columns:

        pivot_auc = (
            backtest.pivot(
                index="test_season",
                columns="model",
                values="roc_auc"
            )
        )

        st.line_chart(
            pivot_auc
        )


# ============================================================
# TAB 5 — MODEL INSIGHTS
# ============================================================

with tab5:

    st.subheader("Model Insights")

    st.metric(
        "Selected Final Model",
        "Random Forest"
    )

    rf_row = summary[
        summary["model"]
        == "Random Forest"
    ]

    if len(rf_row):

        row = rf_row.iloc[0]

        a, b, c = st.columns(3)

        a.metric(
            "ROC-AUC",
            f"{row['roc_auc']:.3f}"
        )

        b.metric(
            "Average Precision",
            f"{row['average_precision']:.3f}"
        )

        c.metric(
            "Precision@20",
            f"{row['precision_at_20']:.1%}"
        )

    st.divider()

    st.subheader(
        "Features Used"
    )

    st.write(
        """
        The model uses player age, playing time, goals,
        assists, goals per 90, assists per 90, historical
        market value, position, sub-position and competition.
        """
    )

    st.subheader(
        "Why Walk-Forward Validation?"
    )

    st.write(
        """
        Football scouting is inherently time-dependent.
        Historical predictions must use information available
        before the prediction date. Walk-forward validation
        reduces the risk of future information leaking into training.
        """
    )

    st.subheader(
        "Important Interpretation"
    )

    st.warning(
        "A high model probability means the player resembles "
        "historical players who met the project's breakout definition. "
        "It does not mean the player is guaranteed to transfer or succeed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data source: Transfermarkt-derived public dataset. "
    "Market values and performance have different observation dates; "
    "the dashboard should be interpreted using the documented data cutoff."
)
