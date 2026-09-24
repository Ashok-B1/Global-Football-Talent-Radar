import duckdb
import pandas as pd

DB = r"C:\Users\banot\Downloads\transfermarkt-datasets.duckdb"

con = duckdb.connect(DB, read_only=True)

# ============================================================
# EMERGING / FEEDER LEAGUES AVAILABLE IN THIS DATASET
# ============================================================

feeder_leagues = {
    "BE1": "Belgium",
    "NL1": "Netherlands",
    "PO1": "Portugal",
    "A1": "Austria",
    "DK1": "Denmark",
    "NO1": "Norway",
    "SE1": "Sweden",
    "C1": "Switzerland",
    "KR1": "Croatia",
    "SER1": "Serbia",
    "BRA1": "Brazil",
    "PL1": "Poland",
    "RO1": "Romania",
    "GR1": "Greece",
    "TR1": "Türkiye",
}

top5 = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
}

feeder_ids = ",".join([f"'{x}'" for x in feeder_leagues])
top5_ids = ",".join([f"'{x}'" for x in top5])


# ============================================================
# 1. PLAYER-SEASON DOMESTIC LEAGUE PERFORMANCE
# ============================================================

print("\nBuilding clean domestic player-season dataset...")

query = f"""
WITH base AS (
    SELECT
        a.player_id,
        a.player_club_id AS club_id,
        a.competition_id,
        a.date,
        a.goals,
        a.assists,
        a.minutes_played,

        CASE
            WHEN EXTRACT(MONTH FROM a.date) >= 7
                THEN CAST(EXTRACT(YEAR FROM a.date) AS INTEGER)
            ELSE CAST(EXTRACT(YEAR FROM a.date) AS INTEGER) - 1
        END AS season_start

    FROM appearances a
    WHERE a.competition_id IN ({feeder_ids})
      AND a.minutes_played IS NOT NULL
),

season_perf AS (
    SELECT
        player_id,
        club_id,
        competition_id,
        season_start,
        SUM(goals) AS goals,
        SUM(assists) AS assists,
        SUM(minutes_played) AS minutes
    FROM base
    GROUP BY 1,2,3,4
),

enriched AS (
    SELECT
        s.*,
        p.name AS player_name,
        p.date_of_birth,
        p.position,
        p.sub_position,
        c.name AS club_name,
        c.domestic_competition_id,
        comp.name AS competition_name,
        comp.country_name,

        MAKE_DATE(s.season_start + 1, 6, 30) AS season_end

    FROM season_perf s

    LEFT JOIN players p
        ON s.player_id = p.player_id

    LEFT JOIN clubs c
        ON CAST(s.club_id AS VARCHAR) = c.club_id

    LEFT JOIN competitions comp
        ON s.competition_id = comp.competition_id
)

SELECT
    e.*,

    DATE_DIFF(
        'year',
        CAST(e.date_of_birth AS DATE),
        e.season_end
    ) AS age_at_season_end,

    ROUND(
        CASE
            WHEN e.minutes > 0
            THEN e.goals * 90.0 / e.minutes
            ELSE 0
        END, 3
    ) AS goals_per90,

    ROUND(
        CASE
            WHEN e.minutes > 0
            THEN e.assists * 90.0 / e.minutes
            ELSE 0
        END, 3
    ) AS assists_per90

FROM enriched e
WHERE e.minutes >= 450
"""

df = con.sql(query).df()

print(f"Rows: {len(df):,}")
print(f"Players: {df['player_id'].nunique():,}")
print(f"Seasons: {df['season_start'].nunique()}")
print(f"Leagues: {df['competition_id'].nunique()}")

print("\n=== FEEDER LEAGUE COVERAGE ===")
print(
    df.groupby(["competition_id", "competition_name"])
      .agg(
          players=("player_id", "nunique"),
          player_seasons=("player_id", "size"),
          total_minutes=("minutes", "sum")
      )
      .sort_values("player_seasons", ascending=False)
      .to_string()
)


# ============================================================
# 2. HISTORICAL MARKET VALUE — AS OF SEASON END
#    IMPORTANT: NOT CURRENT VALUE
# ============================================================

print("\nAttaching historical market value as of season end...")

valuation_query = """
SELECT
    player_id,
    date AS valuation_date,
    market_value_in_eur
FROM player_valuations
"""

vals = con.sql(valuation_query).df()
vals["valuation_date"] = pd.to_datetime(vals["valuation_date"])

df["season_end"] = pd.to_datetime(df["season_end"])

# Merge only valuations available by that season's end.
merged = df.merge(
    vals,
    on="player_id",
    how="left"
)

merged = merged[
    merged["valuation_date"] <= merged["season_end"]
].copy()

# Select latest known value available at the end of each player-season.
merged = (
    merged.sort_values(
        ["player_id", "season_start", "valuation_date"]
    )
    .groupby(
        ["player_id", "club_id", "competition_id", "season_start"],
        as_index=False
    )
    .tail(1)
)

# Restore rows that had no valuation record.
keys = [
    "player_id",
    "club_id",
    "competition_id",
    "season_start"
]

base_keys = df[keys].drop_duplicates()

merged_keys = merged[keys].drop_duplicates()

missing = base_keys.merge(
    merged_keys,
    on=keys,
    how="left",
    indicator=True
)

missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")

if len(missing) > 0:
    missing_rows = df.merge(
        missing,
        on=keys,
        how="inner"
    )
    missing_rows["valuation_date"] = pd.NaT
    missing_rows["market_value_in_eur"] = pd.NA
    merged = pd.concat(
        [merged, missing_rows],
        ignore_index=True
    )

# ============================================================
# 3. HISTORICAL TRANSFER / BREAKOUT LABEL
# ============================================================

print("\nCreating historical breakout labels...")

transfers = con.sql(f"""
SELECT
    t.player_id,
    t.transfer_date,
    t.from_club_id,
    t.to_club_id,
    t.transfer_fee,
    t.market_value_in_eur AS transfer_market_value,
    c_to.domestic_competition_id AS destination_competition
FROM transfers t
LEFT JOIN clubs c_to
    ON CAST(t.to_club_id AS VARCHAR) = c_to.club_id
WHERE t.transfer_date IS NOT NULL
  AND t.transfer_date <= DATE '2026-06-30'
  AND c_to.domestic_competition_id IN ({top5_ids})
""").df()

transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])

# A player-season is a positive breakout example when:
# 1. The player was in one of our feeder leagues.
# 2. After that season, they transferred to a Top-5 league.
# 3. The transfer occurred within 3 years after that season.
#
# We also require the transfer to originate from the player's
# tracked club where possible.

perf = merged.copy()
perf["season_end"] = pd.to_datetime(perf["season_end"])

label_query = """
SELECT
    p.*,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM transfers t
            WHERE t.player_id = p.player_id
              AND t.transfer_date > p.season_end
              AND t.transfer_date <= p.season_end + INTERVAL '3 years'
              AND t.from_club_id = p.club_id
        )
        THEN 1
        ELSE 0
    END AS breakout_label
FROM perf AS p
"""

con.register("perf", perf)
con.register("transfers", transfers)

labeled = con.sql(label_query).df()

# ============================================================
# 4. CURRENT TALENT RADAR
# ============================================================

print("\nBuilding current talent radar...")

current = labeled[
    labeled["season_start"] == labeled["season_start"].max()
].copy()

# Current/most recent market value, but used ONLY for the current radar.
current_vals = (
    vals.sort_values(["player_id", "valuation_date"])
        .groupby("player_id", as_index=False)
        .tail(1)
)

current = current.drop(
    columns=["market_value_in_eur"],
    errors="ignore"
)

current = current.merge(
    current_vals[
        ["player_id", "valuation_date", "market_value_in_eur"]
    ],
    on="player_id",
    how="left"
)

# ============================================================
# 5. SAVE
# ============================================================

labeled_file = "player_season_breakout.csv"
current_file = "current_talent_radar_base.csv"

labeled.to_csv(labeled_file, index=False)
current.to_csv(current_file, index=False)

print("\n==========================================")
print("BUILD COMPLETE")
print("==========================================")

print(f"\nHistorical dataset : {labeled_file}")
print(f"Rows               : {len(labeled):,}")
print(f"Players            : {labeled['player_id'].nunique():,}")
print(f"Positive labels    : {labeled['breakout_label'].sum():,}")
print(
    f"Breakout rate      : "
    f"{labeled['breakout_label'].mean() * 100:.2f}%"
)

print(f"\nCurrent radar      : {current_file}")
print(f"Current players    : {len(current):,}")

print("\n=== BREAKOUT LABEL BY LEAGUE ===")
print(
    labeled.groupby("competition_name")["breakout_label"]
           .agg(["count", "sum", "mean"])
           .sort_values("mean", ascending=False)
           .to_string()
)

con.close()
print("\nDONE")