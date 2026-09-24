import pandas as pd
import numpy as np

INPUT = "current_talent_radar.csv"

df = pd.read_csv(INPUT)

# ============================================================
# CLEAN CURRENT DATA
# ============================================================

df["market_value_in_eur"] = pd.to_numeric(
    df["market_value_in_eur"],
    errors="coerce"
)

df["age_at_season_end"] = pd.to_numeric(
    df["age_at_season_end"],
    errors="coerce"
)

df["minutes"] = pd.to_numeric(
    df["minutes"],
    errors="coerce"
)

df["breakout_probability_pct"] = pd.to_numeric(
    df["breakout_probability_pct"],
    errors="coerce"
)

# ============================================================
# EMERGING TALENT FILTER
# ============================================================

candidates = df[
    (df["age_at_season_end"] <= 23)
    & (df["market_value_in_eur"] <= 10_000_000)
    & (df["minutes"] >= 900)
    & (df["breakout_probability_pct"].notna())
].copy()

# Rank primarily by model breakout probability.
# Market value and age are shown for scouting context.
candidates = candidates.sort_values(
    [
        "breakout_probability_pct",
        "market_value_in_eur",
        "age_at_season_end"
    ],
    ascending=[False, True, True]
).reset_index(drop=True)

candidates["scouting_rank"] = (
    np.arange(len(candidates)) + 1
)

# ============================================================
# ADD READABLE VALUE
# ============================================================

candidates["market_value_m"] = (
    candidates["market_value_in_eur"] / 1_000_000
).round(2)

# ============================================================
# OUTPUT
# ============================================================

columns = [
    "scouting_rank",
    "player_name",
    "club_name",
    "competition_name",
    "country_name",
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

final_radar = candidates[columns]

final_radar.to_csv(
    "final_talent_radar.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("EMERGING TALENT RADAR")
print("============================================")

print(f"All current players: {len(df):,}")
print(f"Players meeting scouting filter: {len(candidates):,}")

print("\nFilter:")
print("Age <= 23")
print("Market value <= €10M")
print("Minutes >= 900")

print("\n=== TOP 30 EMERGING TALENT CANDIDATES ===")

print(
    final_radar.head(30).to_string(index=False)
)

print("\n=== CANDIDATES BY LEAGUE ===")

print(
    candidates.groupby("competition_name")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

print("\nSaved: final_talent_radar.csv")
print("\nDONE")