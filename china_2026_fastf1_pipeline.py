import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd
from fastf1 import RateLimitExceededError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------------
# FastF1 cache setup (critical)
# -------------------------------
CACHE_DIR = Path(__file__).resolve().parent / "f1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))
OUT_DIR = Path(__file__).resolve().parent
TRAINING_DATASET_PATH = OUT_DIR / "china_training_dataset_2023_2025.csv"
QUALI_OUTPUT_PATH = OUT_DIR / "china_2026_predicted_qualifying_top10.csv"
RACE_OUTPUT_PATH = OUT_DIR / "china_2026_predicted_race_order.csv"

# Keep API usage under control while preserving track-specific signal.
MAX_EVENTS_PER_YEAR = 6
DEFAULT_USE_TELEMETRY = False


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert timedelta-like pandas series to float seconds."""
    return series.dt.total_seconds()


def _mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else np.nan


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    cols = ["Driver", "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Compound", "Stint"]
    use_cols = [c for c in cols if c in laps.columns]
    out = laps[use_cols].copy()

    # Keep only timed laps with sector data to stabilize pace features.
    out = out.dropna(subset=[c for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"] if c in out.columns])
    if "LapTime" in out.columns:
        out["LapTime_s"] = _to_seconds(out["LapTime"])
    if "Sector1Time" in out.columns:
        out["Sector1_s"] = _to_seconds(out["Sector1Time"])
    if "Sector2Time" in out.columns:
        out["Sector2_s"] = _to_seconds(out["Sector2Time"])
    if "Sector3Time" in out.columns:
        out["Sector3_s"] = _to_seconds(out["Sector3Time"])

    return out


def _driver_degradation_s_per_lap(driver_laps: pd.DataFrame) -> float:
    """Estimate tire degradation from lap time slope within each stint."""
    if "Stint" not in driver_laps.columns or "LapNumber" not in driver_laps.columns:
        return np.nan

    slopes: List[float] = []
    for _, stint_df in driver_laps.groupby("Stint"):
        stint_df = stint_df.dropna(subset=["LapTime_s", "LapNumber"]).sort_values("LapNumber")
        if len(stint_df) < 4:
            continue
        x = stint_df["LapNumber"].to_numpy(dtype=float)
        y = stint_df["LapTime_s"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, deg=1)[0]
        slopes.append(float(slope))

    if not slopes:
        return np.nan

    return float(np.mean(slopes))


def _session_weather_features(session) -> Dict[str, float]:
    """Aggregate FastF1 weather measurements for session-level weather features."""
    weather = getattr(session, "weather_data", None)
    if weather is None or weather.empty:
        return {
            "weather_airtemp": np.nan,
            "weather_tracktemp": np.nan,
            "weather_humidity": np.nan,
            "weather_wind_speed": np.nan,
            "weather_rainfall": np.nan,
        }

    return {
        "weather_airtemp": _mean_or_nan(_safe_numeric(weather.get("AirTemp", pd.Series(dtype=float)))),
        "weather_tracktemp": _mean_or_nan(_safe_numeric(weather.get("TrackTemp", pd.Series(dtype=float)))),
        "weather_humidity": _mean_or_nan(_safe_numeric(weather.get("Humidity", pd.Series(dtype=float)))),
        "weather_wind_speed": _mean_or_nan(_safe_numeric(weather.get("WindSpeed", pd.Series(dtype=float)))),
        "weather_rainfall": _mean_or_nan(_safe_numeric(weather.get("Rainfall", pd.Series(dtype=float)))),
    }


def _event_name_from_schedule(schedule_row: pd.Series) -> str:
    for key in ["EventName", "OfficialEventName", "Country"]:
        if key in schedule_row and pd.notna(schedule_row[key]):
            return str(schedule_row[key])
    return "Unknown"


def _is_china_event_name(name: str) -> bool:
    lower = name.lower()
    return "china" in lower or "chinese" in lower or "shanghai" in lower


def _select_events_for_year(schedule: pd.DataFrame, max_events_per_year: Optional[int]) -> pd.DataFrame:
    """
    Select a bounded set of events per season to avoid API rate-limit failures.
    Always keep Chinese GP events and then append the most recent rounds.
    """
    if schedule.empty:
        return schedule

    if "RoundNumber" in schedule.columns:
        schedule = schedule.sort_values("RoundNumber").reset_index(drop=True)

    if max_events_per_year is None or max_events_per_year <= 0:
        return schedule

    event_names = schedule.apply(_event_name_from_schedule, axis=1)
    china_mask = event_names.apply(_is_china_event_name)

    china_events = schedule[china_mask]
    other_events = schedule[~china_mask]

    keep_other_count = max(0, max_events_per_year - len(china_events))
    recent_other_events = other_events.tail(keep_other_count)

    selected = pd.concat([china_events, recent_other_events], ignore_index=False)
    selected = selected.drop_duplicates()
    if "RoundNumber" in selected.columns:
        selected = selected.sort_values("RoundNumber")

    return selected.reset_index(drop=True)


def _load_session(year: int, event_name: str, session_code: str, use_telemetry: bool = DEFAULT_USE_TELEMETRY):
    session = fastf1.get_session(year, event_name, session_code)
    session.load(telemetry=use_telemetry, weather=True, laps=True, messages=False)
    return session


def _build_event_driver_rows(
    year: int,
    round_number: int,
    event_name: str,
    use_telemetry: bool = DEFAULT_USE_TELEMETRY,
) -> pd.DataFrame:
    """
    Build one row per driver for a race weekend using qualifying and race session data.
    """
    quali = _load_session(year, event_name, "Q", use_telemetry=use_telemetry)
    race = _load_session(year, event_name, "R", use_telemetry=use_telemetry)

    q_results = quali.results.copy()
    r_results = race.results.copy()

    q_laps = _clean_laps(quali.laps)
    r_laps = _clean_laps(race.laps)

    q_weather = _session_weather_features(quali)
    r_weather = _session_weather_features(race)

    q_driver_feats = []
    if not q_laps.empty:
        for drv, drv_laps in q_laps.groupby("Driver"):
            q_driver_feats.append(
                {
                    "Driver": drv,
                    "q_best_lap_s": float(drv_laps["LapTime_s"].min()),
                    "q_mean_lap_s": float(drv_laps["LapTime_s"].mean()),
                    "q_s1_s": float(drv_laps["Sector1_s"].mean()),
                    "q_s2_s": float(drv_laps["Sector2_s"].mean()),
                    "q_s3_s": float(drv_laps["Sector3_s"].mean()),
                }
            )
    q_driver_feats_df = pd.DataFrame(q_driver_feats)

    r_driver_feats = []
    if not r_laps.empty:
        for drv, drv_laps in r_laps.groupby("Driver"):
            r_driver_feats.append(
                {
                    "Driver": drv,
                    "r_mean_lap_s": float(drv_laps["LapTime_s"].mean()),
                    "r_median_lap_s": float(drv_laps["LapTime_s"].median()),
                    "r_best_lap_s": float(drv_laps["LapTime_s"].min()),
                    "r_s1_s": float(drv_laps["Sector1_s"].mean()),
                    "r_s2_s": float(drv_laps["Sector2_s"].mean()),
                    "r_s3_s": float(drv_laps["Sector3_s"].mean()),
                    "r_total_sector_s": float(drv_laps[["Sector1_s", "Sector2_s", "Sector3_s"]].mean().sum()),
                    "r_stint_count": float(drv_laps["Stint"].nunique()) if "Stint" in drv_laps.columns else np.nan,
                    "r_compound_count": float(drv_laps["Compound"].nunique()) if "Compound" in drv_laps.columns else np.nan,
                    "r_tyre_deg_s_per_lap": _driver_degradation_s_per_lap(drv_laps),
                }
            )
    r_driver_feats_df = pd.DataFrame(r_driver_feats)

    base = r_results[["Abbreviation", "FullName", "TeamName", "Position", "GridPosition", "Points", "Status"]].copy()
    base = base.rename(
        columns={
            "Abbreviation": "Driver",
            "FullName": "DriverName",
            "TeamName": "Team",
            "Position": "race_finish_position",
            "GridPosition": "grid_position",
            "Points": "race_points",
            "Status": "race_status",
        }
    )

    q_cols = ["Abbreviation", "Position", "Q1", "Q2", "Q3"]
    q_cols = [c for c in q_cols if c in q_results.columns]
    q_base = q_results[q_cols].copy()
    q_base = q_base.rename(columns={"Abbreviation": "Driver", "Position": "qualifying_position"})

    for q_seg in ["Q1", "Q2", "Q3"]:
        if q_seg in q_base.columns:
            q_base[f"{q_seg.lower()}_s"] = _to_seconds(q_base[q_seg])

    out = base.merge(q_base[[c for c in q_base.columns if c not in ["Q1", "Q2", "Q3"]]], on="Driver", how="left")
    out = out.merge(q_driver_feats_df, on="Driver", how="left")
    out = out.merge(r_driver_feats_df, on="Driver", how="left")

    out["year"] = year
    out["round"] = round_number
    out["event_name"] = event_name
    out["is_china"] = 1.0 if "china" in event_name.lower() or "chinese" in event_name.lower() else 0.0

    for k, v in q_weather.items():
        out[f"q_{k}"] = v
    for k, v in r_weather.items():
        out[f"r_{k}"] = v

    return out


def build_training_dataset(
    years: List[int],
    max_events_per_year: Optional[int] = MAX_EVENTS_PER_YEAR,
    use_telemetry: bool = DEFAULT_USE_TELEMETRY,
) -> pd.DataFrame:
    """
    Pull all race weekends from selected seasons and build model-ready rows.
    """
    rows = []
    for year in years:
        print(f"Loading season {year}...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except RateLimitExceededError as exc:
            print(f"  -> Skipped season {year}: {exc}")
            continue

        # Keep only race weekends (exclude pre-season and special events).
        if "EventFormat" in schedule.columns:
            schedule = schedule[schedule["EventFormat"].isin(["conventional", "sprint", "sprint_shootout", "sprint_qualifying"])]

        schedule = _select_events_for_year(schedule, max_events_per_year)

        for _, event in schedule.iterrows():
            event_name = _event_name_from_schedule(event)
            round_number = int(event.get("RoundNumber", 0))
            print(f"  -> {year} Round {round_number}: {event_name}")
            try:
                rows.append(_build_event_driver_rows(year, round_number, event_name, use_telemetry=use_telemetry))
            except RateLimitExceededError as exc:
                print(f"     Rate limit reached at {year} {event_name}: {exc}")
                print("     Continuing with data gathered so far.")
                break
            except Exception as exc:
                print(f"     Skipped {year} {event_name}: {exc}")

    if not rows:
        raise RuntimeError("No training rows collected. Check FastF1 access and cache path.")

    df = pd.concat(rows, ignore_index=True)

    # Standardize numeric targets and inputs.
    numeric_cols = [
        "qualifying_position",
        "race_finish_position",
        "grid_position",
        "race_points",
        "q_best_lap_s",
        "q_mean_lap_s",
        "q_s1_s",
        "q_s2_s",
        "q_s3_s",
        "r_mean_lap_s",
        "r_median_lap_s",
        "r_best_lap_s",
        "r_s1_s",
        "r_s2_s",
        "r_s3_s",
        "r_total_sector_s",
        "r_stint_count",
        "r_compound_count",
        "r_tyre_deg_s_per_lap",
        "q1_s",
        "q2_s",
        "q3_s",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rolling form features to capture recent performance from 2023-2025.
    df = df.sort_values(["Driver", "year", "round"]).reset_index(drop=True)
    df["driver_form_finish_mean_5"] = (
        df.groupby("Driver")["race_finish_position"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df["driver_form_points_mean_5"] = (
        df.groupby("Driver")["race_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df = df.sort_values(["Team", "year", "round"]).reset_index(drop=True)
    df["team_form_points_mean_5"] = (
        df.groupby("Team")["race_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Shanghai-specific history from previous years for each driver.
    china_hist = (
        df[df["is_china"] == 1.0]
        .groupby("Driver")[["q_best_lap_s", "r_mean_lap_s", "r_tyre_deg_s_per_lap", "qualifying_position", "race_finish_position"]]
        .mean()
        .rename(
            columns={
                "q_best_lap_s": "china_hist_q_best_lap_s",
                "r_mean_lap_s": "china_hist_r_mean_lap_s",
                "r_tyre_deg_s_per_lap": "china_hist_tyre_deg_s_per_lap",
                "qualifying_position": "china_hist_qualifying_position",
                "race_finish_position": "china_hist_race_finish_position",
            }
        )
        .reset_index()
    )

    df = df.merge(china_hist, on="Driver", how="left")

    return df


def save_training_dataset(df: pd.DataFrame, output_path: Path = TRAINING_DATASET_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_training_dataset(input_path: Path = TRAINING_DATASET_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {input_path}. "
            "Run phase 'prepare' first to extract and save data."
        )
    return pd.read_csv(input_path)


def _build_model(feature_df: pd.DataFrame, model_type: str = "ensemble") -> Pipeline:
    numeric_features = [c for c in feature_df.columns if c not in ["Driver", "DriverName", "Team", "event_name"]]
    categorical_features = [c for c in ["Driver", "Team", "event_name"] if c in feature_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )

    if model_type == "rf":
        reg = RandomForestRegressor(n_estimators=400, random_state=42)
    elif model_type == "gbr":
        reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42)
    else:
        reg = VotingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
                ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
            ]
        )

    model = Pipeline(steps=[("prep", preprocessor), ("reg", reg)])
    return model


def train_models(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline, Dict[str, float], List[str], List[str]]:
    """
    Train separate models for:
    1) Qualifying position prediction
    2) Race finish prediction (uses predicted qualifying as an input during inference)
    """
    base_feature_cols = [
        "Driver",
        "DriverName",
        "Team",
        "event_name",
        "year",
        "round",
        "is_china",
        "driver_form_finish_mean_5",
        "driver_form_points_mean_5",
        "team_form_points_mean_5",
        "q_best_lap_s",
        "q_mean_lap_s",
        "q_s1_s",
        "q_s2_s",
        "q_s3_s",
        "r_mean_lap_s",
        "r_median_lap_s",
        "r_best_lap_s",
        "r_s1_s",
        "r_s2_s",
        "r_s3_s",
        "r_total_sector_s",
        "r_stint_count",
        "r_compound_count",
        "r_tyre_deg_s_per_lap",
        "q_weather_airtemp",
        "q_weather_tracktemp",
        "q_weather_humidity",
        "q_weather_wind_speed",
        "q_weather_rainfall",
        "r_weather_airtemp",
        "r_weather_tracktemp",
        "r_weather_humidity",
        "r_weather_wind_speed",
        "r_weather_rainfall",
        "china_hist_q_best_lap_s",
        "china_hist_r_mean_lap_s",
        "china_hist_tyre_deg_s_per_lap",
        "china_hist_qualifying_position",
        "china_hist_race_finish_position",
    ]

    available_base_cols = [c for c in base_feature_cols if c in df.columns]

    # Qualifying model target.
    quali_df = df.dropna(subset=["qualifying_position"]).copy()
    X_q = quali_df[available_base_cols].copy()
    y_q = quali_df["qualifying_position"].astype(float)

    Xq_train, Xq_test, yq_train, yq_test = train_test_split(X_q, y_q, test_size=0.2, random_state=42)
    quali_model = _build_model(X_q, model_type="ensemble")
    quali_model.fit(Xq_train, yq_train)
    q_pred = quali_model.predict(Xq_test)
    quali_mae = float(mean_absolute_error(yq_test, q_pred))

    # Race model uses qualifying position as a direct pace/order signal.
    race_feature_cols = available_base_cols + ["qualifying_position"]
    race_df = df.dropna(subset=["race_finish_position"]).copy()
    X_r = race_df[race_feature_cols].copy()
    y_r = race_df["race_finish_position"].astype(float)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    race_model = _build_model(X_r, model_type="ensemble")
    race_model.fit(Xr_train, yr_train)
    r_pred = race_model.predict(Xr_test)
    race_mae = float(mean_absolute_error(yr_test, r_pred))

    metrics = {
        "qualifying_mae_positions": quali_mae,
        "race_mae_positions": race_mae,
        "training_rows": float(len(df)),
    }

    return quali_model, race_model, metrics, available_base_cols, race_feature_cols


def _latest_driver_state(df: pd.DataFrame, max_year: int = 2026) -> pd.DataFrame:
    """Pick latest available event row per driver as baseline for 2026 prediction."""
    subset = df[df["year"] <= max_year].copy()
    subset = subset.sort_values(["Driver", "year", "round"]).groupby("Driver", as_index=False).tail(1)
    return subset


def build_2026_china_inference_frame(df: pd.DataFrame, base_feature_cols: List[str]) -> pd.DataFrame:
    """
    Create inference rows for upcoming 2026 Chinese GP using latest known form
    and Shanghai historical aggregates.
    """
    latest = _latest_driver_state(df, max_year=2026).copy()
    latest["year"] = 2026
    latest["round"] = 2  # China is expected as early-season round in 2026.
    latest["event_name"] = "Chinese Grand Prix"
    latest["is_china"] = 1.0

    # If weather for 2026 is not yet available, use China historical race-weekend means.
    china_weather_means = (
        df[df["is_china"] == 1.0][
            [
                "q_weather_airtemp",
                "q_weather_tracktemp",
                "q_weather_humidity",
                "q_weather_wind_speed",
                "q_weather_rainfall",
                "r_weather_airtemp",
                "r_weather_tracktemp",
                "r_weather_humidity",
                "r_weather_wind_speed",
                "r_weather_rainfall",
            ]
        ]
        .mean(numeric_only=True)
        .to_dict()
    )
    for col, val in china_weather_means.items():
        if col in latest.columns:
            latest[col] = val

    # Overwrite previous race's track-dependent pace values with Shanghai-specific history.
    latest["q_best_lap_s"] = latest.get("china_hist_q_best_lap_s", np.nan)
    latest["r_mean_lap_s"] = latest.get("china_hist_r_mean_lap_s", np.nan)
    latest["r_tyre_deg_s_per_lap"] = latest.get("china_hist_tyre_deg_s_per_lap", np.nan)

    # No reliable Shanghai sector split history in this block; let imputer handle these.
    for c in ["q_s1_s", "q_s2_s", "q_s3_s", "r_s1_s", "r_s2_s", "r_s3_s", "q_mean_lap_s"]:
        if c in latest.columns:
            latest[c] = np.nan

    return latest[base_feature_cols].copy()


def predict_2026_china(
    quali_model: Pipeline,
    race_model: Pipeline,
    inference_base: pd.DataFrame,
    race_feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict 2026 Chinese GP qualifying order and race result.
    """
    output = inference_base[[c for c in ["Driver", "DriverName", "Team"] if c in inference_base.columns]].copy()

    # Predict qualifying positions (lower is better).
    pred_quali_pos = quali_model.predict(inference_base)
    output["predicted_qualifying_position"] = pred_quali_pos
    quali_top10 = output.sort_values("predicted_qualifying_position").head(10).reset_index(drop=True)
    quali_top10.index = quali_top10.index + 1
    quali_top10["PredictedGrid"] = quali_top10.index

    # Race model receives predicted qualifying as an input feature.
    race_input = inference_base.copy()
    race_input["qualifying_position"] = pred_quali_pos
    race_input = race_input.reindex(columns=race_feature_cols)

    pred_race_finish = race_model.predict(race_input)
    output["predicted_race_finish_position"] = pred_race_finish
    race_order = output.sort_values("predicted_race_finish_position").reset_index(drop=True)
    race_order.index = race_order.index + 1
    race_order["PredictedRacePosition"] = race_order.index

    return quali_top10, race_order


def run_prepare_phase(
    years: List[int],
    max_events_per_year: Optional[int],
    use_telemetry: bool = DEFAULT_USE_TELEMETRY,
) -> pd.DataFrame:
    print("Phase 1/2: Preparing dataset from FastF1 (cache + feature extraction)...")
    if 2023 in years:
        print("Note: Chinese GP was not held in 2023; Shanghai-specific history comes from available years.")

    if use_telemetry:
        print("Telemetry mode enabled: sector completeness may improve but downloads will be larger.")

    df = build_training_dataset(
        years,
        max_events_per_year=max_events_per_year,
        use_telemetry=use_telemetry,
    )
    save_training_dataset(df, TRAINING_DATASET_PATH)
    print(f"Saved training dataset: {TRAINING_DATASET_PATH.name}")
    print(f"Dataset rows: {len(df)} | Drivers: {df['Driver'].nunique()}")
    return df


def run_train_phase(df: pd.DataFrame) -> None:
    print("Phase 2/2: Training models and generating predictions...")
    print(f"Dataset ready with {len(df)} rows and {df['Driver'].nunique()} drivers.")

    print("Training qualifying and race models...")
    quali_model, race_model, metrics, base_feature_cols, race_feature_cols = train_models(df)

    print("\nModel Evaluation")
    print(f"Qualifying MAE (positions): {metrics['qualifying_mae_positions']:.3f}")
    print(f"Race MAE (positions): {metrics['race_mae_positions']:.3f}")

    print("\nPreparing 2026 Chinese GP inference frame...")
    inference_base = build_2026_china_inference_frame(df, base_feature_cols)

    quali_top10, race_order = predict_2026_china(quali_model, race_model, inference_base, race_feature_cols)

    print("\nAnalyzing Shanghai-specific feature importances...")
    china_data = df.dropna(subset=["race_finish_position"]).copy()
    china_data = china_data[china_data["is_china"] == 1.0]

    if not china_data.empty and len(china_data) > 0:
        X_china = china_data[race_feature_cols].copy()
        y_china = china_data["race_finish_position"].astype(float)
        
        from sklearn.inspection import permutation_importance
        result = permutation_importance(race_model, X_china, y_china, n_repeats=10, random_state=42)
        importances = pd.DataFrame(
            {"feature": race_feature_cols, "importance": result.importances_mean}
        ).sort_values("importance", ascending=False)
        
        print("Top 10 features indicative of race success at Shanghai (Permutation Importance):")
        for idx, row in importances.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("Not enough Shanghai data to compute specific feature importances.")

    print("\nPredicted 2026 Chinese GP Qualifying Top 10")
    print(quali_top10[["PredictedGrid", "Driver", "DriverName", "Team", "predicted_qualifying_position"]])

    predicted_winner = race_order.iloc[0]
    print("\nPredicted 2026 Chinese GP Race Winner")
    print(
        f"P1: {predicted_winner['Driver']} | "
        f"{predicted_winner.get('DriverName', 'Unknown')} | "
        f"Team: {predicted_winner.get('Team', 'Unknown')} | "
        f"Predicted finish position score: {predicted_winner['predicted_race_finish_position']:.3f}"
    )

    race_top3 = race_order.head(3).reset_index(drop=True)
    print("\nPredicted 2026 Chinese GP Race Podium (Top 3)")
    for idx, row in race_top3.iterrows():
        pos = idx + 1
        print(
            f"P{pos}: {row['Driver']} | "
            f"{row.get('DriverName', 'Unknown')} | "
            f"Team: {row.get('Team', 'Unknown')} | "
            f"Predicted finish position score: {row['predicted_race_finish_position']:.3f}"
        )

    quali_top10.to_csv(QUALI_OUTPUT_PATH, index=False)
    race_order.to_csv(RACE_OUTPUT_PATH, index=False)

    print("\nSaved outputs:")
    print(f"- {TRAINING_DATASET_PATH.name}")
    print(f"- {QUALI_OUTPUT_PATH.name}")
    print(f"- {RACE_OUTPUT_PATH.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-phase FastF1 pipeline for 2026 Chinese GP prediction. "
            "Phase 1 prepares and saves dataset, Phase 2 trains and predicts from saved dataset."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["all", "prepare", "train"],
        default="all",
        help="Execution phase: all (prepare+train), prepare only, or train only.",
    )
    parser.add_argument(
        "--max-events-per-year",
        type=int,
        default=MAX_EVENTS_PER_YEAR,
        help="Maximum events per season to fetch during prepare phase (<=0 means all).",
    )
    parser.add_argument(
        "--use-telemetry",
        action="store_true",
        help="Enable FastF1 telemetry download during prepare phase for potentially better sector completeness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = [2023, 2024, 2025, 2026]

    if args.phase == "prepare":
        run_prepare_phase(
            years,
            max_events_per_year=args.max_events_per_year,
            use_telemetry=args.use_telemetry,
        )
        return

    if args.phase == "train":
        print("Loading prebuilt dataset from disk for train-only mode...")
        df = load_training_dataset(TRAINING_DATASET_PATH)
        run_train_phase(df)
        return

    df = run_prepare_phase(
        years,
        max_events_per_year=args.max_events_per_year,
        use_telemetry=args.use_telemetry,
    )
    run_train_phase(df)


if __name__ == "__main__":
    main()
