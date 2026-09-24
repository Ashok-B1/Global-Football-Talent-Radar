import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "player_season_breakout.csv"
CURRENT_FILE = "current_talent_radar_base.csv"

DATA_CUTOFF = pd.Timestamp("2026-06-30")

# 3-year outcome window means a season can only be labelled
# if its complete future window exists by the data cutoff.
MAX_FULLY_OBSERVED_SEASON = 2022


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading historical data...")

df = pd.read_csv(DATA_FILE)
current = pd.read_csv(CURRENT_FILE)

df["season_start"] = pd.to_numeric(
    df["season_start"], errors="coerce"
)

df["season_end"] = pd.to_datetime(
    df["season_end"], errors="coerce"
)

current["season_start"] = pd.to_numeric(
    current["season_start"], errors="coerce"
)

print(f"Historical rows: {len(df):,}")
print(f"Historical players: {df['player_id'].nunique():,}")
print(f"Current rows: {len(current):,}")


# ============================================================
# FIX CENSORED OUTCOMES
# ============================================================

print("\nFiltering to fully observed 3-year breakout windows...")

model_df = df[
    df["season_start"].notna()
    & (df["season_start"] <= MAX_FULLY_OBSERVED_SEASON)
].copy()

print(
    f"Training/backtest rows: {len(model_df):,}"
)

print(
    f"Seasons covered: "
    f"{int(model_df['season_start'].min())}/"
    f"{str(int(model_df['season_start'].max()) + 1)[-2:]}"
)

print(
    f"Breakout rate: "
    f"{model_df['breakout_label'].mean() * 100:.2f}%"
)


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "age_at_season_end",
    "minutes",
    "goals",
    "assists",
    "goals_per90",
    "assists_per90",
    "market_value_in_eur",
]

categorical_features = [
    "position",
    "sub_position",
    "competition_id",
]

feature_columns = numeric_features + categorical_features

X = model_df[feature_columns].copy()
y = model_df["breakout_label"].astype(int)

# Log-transform market value separately.
X["log_market_value"] = np.log1p(
    X["market_value_in_eur"].fillna(0)
)

numeric_features_model = numeric_features + [
    "log_market_value"
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features_model),
    ("cat", categorical_pipe, categorical_features),
])


# ============================================================
# MODELS
# ============================================================

logistic = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

random_forest = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
    )
])

models = {
    "Logistic Regression": logistic,
    "Random Forest": random_forest,
}


# ============================================================
# PRECISION @ K
# ============================================================

def precision_at_k(y_true, scores, k):
    n = min(k, len(y_true))

    if n == 0:
        return np.nan

    order = np.argsort(scores)[::-1][:n]

    return float(
        np.mean(np.asarray(y_true)[order])
    )


def recall_at_k(y_true, scores, k):
    total_positive = int(np.sum(y_true))

    if total_positive == 0:
        return np.nan

    n = min(k, len(y_true))

    order = np.argsort(scores)[::-1][:n]

    return float(
        np.sum(np.asarray(y_true)[order])
        / total_positive
    )


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

test_seasons = sorted(
    model_df["season_start"].unique()
)

# Start later so every fold has enough training history.
test_seasons = [
    s for s in test_seasons
    if s >= 2018
]

results = []

print("\n============================================")
print("WALK-FORWARD BACKTEST")
print("============================================")

for test_season in test_seasons:

    train = model_df[
        model_df["season_start"] < test_season
    ].copy()

    test = model_df[
        model_df["season_start"] == test_season
    ].copy()

    if len(train) < 1000 or len(test) < 20:
        continue

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    X_train["log_market_value"] = np.log1p(
        X_train["market_value_in_eur"].fillna(0)
    )

    X_test["log_market_value"] = np.log1p(
        X_test["market_value_in_eur"].fillna(0)
    )

    y_train = train["breakout_label"].astype(int)
    y_test = test["breakout_label"].astype(int)

    print(
        f"\nTest season: {int(test_season)}/"
        f"{str(int(test_season)+1)[-2:]}"
    )

    print(
        f"Train rows: {len(train):,} | "
        f"Test rows: {len(test):,}"
    )

    for name, model in models.items():

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        ap = average_precision_score(
            y_test,
            probabilities
        )

        pred = (
            probabilities >= 0.50
        ).astype(int)

        results.append({
            "test_season": int(test_season),
            "model": name,
            "roc_auc": auc,
            "average_precision": ap,
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
            "precision_at_10": precision_at_k(
                y_test, probabilities, 10
            ),
            "precision_at_20": precision_at_k(
                y_test, probabilities, 20
            ),
            "precision_at_50": precision_at_k(
                y_test, probabilities, 50
            ),
            "recall_at_10": recall_at_k(
                y_test, probabilities, 10
            ),
            "recall_at_20": recall_at_k(
                y_test, probabilities, 20
            ),
            "recall_at_50": recall_at_k(
                y_test, probabilities, 50
            ),
        })

        print(
            f"{name}: "
            f"ROC-AUC={auc:.3f}, "
            f"AP={ap:.3f}, "
            f"P@20={results[-1]['precision_at_20']:.3f}"
        )


# ============================================================
# BACKTEST SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n============================================")
print("BACKTEST SUMMARY")
print("============================================")

summary = (
    results_df
    .groupby("model")
    .agg({
        "roc_auc": "mean",
        "average_precision": "mean",
        "precision": "mean",
        "recall": "mean",
        "f1": "mean",
        "precision_at_10": "mean",
        "precision_at_20": "mean",
        "precision_at_50": "mean",
        "recall_at_10": "mean",
        "recall_at_20": "mean",
        "recall_at_50": "mean",
    })
    .sort_values(
        "precision_at_20",
        ascending=False
    )
)

print(
    summary.round(3).to_string()
)

results_df.to_csv(
    "walk_forward_results.csv",
    index=False
)

summary.to_csv(
    "model_summary.csv"
)


# ============================================================
# SELECT FINAL MODEL
# ============================================================

best_model_name = (
    summary["precision_at_20"]
    .idxmax()
)

best_model = models[
    best_model_name
]

print(
    f"\nSelected final model: {best_model_name}"
)


# ============================================================
# TRAIN FINAL MODEL ON ALL FULLY OBSERVED DATA
# ============================================================

print("\nTraining final model...")

X_all = model_df[feature_columns].copy()

X_all["log_market_value"] = np.log1p(
    X_all["market_value_in_eur"].fillna(0)
)

best_model.fit(X_all, y)


# ============================================================
# CURRENT TALENT RADAR
# ============================================================

print("\nGenerating current Talent Radar...")

current_features = current[
    feature_columns
].copy()

current_features["log_market_value"] = np.log1p(
    current_features[
        "market_value_in_eur"
    ].fillna(0)
)

current["breakout_probability"] = (
    best_model.predict_proba(
        current_features
    )[:, 1]
)

current["breakout_probability_pct"] = (
    current["breakout_probability"] * 100
)

# Remove very low playing-time players
current = current[
    current["minutes"] >= 450
].copy()

# Rank candidates
current = current.sort_values(
    [
        "breakout_probability",
        "market_value_in_eur",
        "minutes"
    ],
    ascending=[False, True, False]
).copy()

current["scouting_rank"] = (
    np.arange(len(current)) + 1
)

# ============================================================
# DISPLAY TOP 50
# ============================================================

radar_columns = [
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
    "market_value_in_eur",
    "breakout_probability_pct",
]

radar = current[
    radar_columns
]

print("\n============================================")
print("TOP 20 CURRENT SCOUTING CANDIDATES")
print("============================================")

print(
    radar.head(20).to_string(index=False)
)

radar.to_csv(
    "current_talent_radar.csv",
    index=False
)


# ============================================================
# DATA QUALITY
# ============================================================

print("\n============================================")
print("DATA QUALITY CHECK")
print("============================================")

print(
    f"Historical players: "
    f"{model_df['player_id'].nunique():,}"
)

print(
    f"Historical rows: "
    f"{len(model_df):,}"
)

print(
    f"Positive breakout examples: "
    f"{int(y.sum()):,}"
)

print(
    f"Breakout rate: "
    f"{y.mean()*100:.2f}%"
)

print(
    f"Current candidates: "
    f"{len(current):,}"
)

print("\nFiles created:")
print("  walk_forward_results.csv")
print("  model_summary.csv")
print("  current_talent_radar.csv")

print("\nDONE.")
