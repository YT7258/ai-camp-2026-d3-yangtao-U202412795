"""Forecast next-hour power from real chronological household measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


LAGS = (1, 2, 3, 24)
EXPECTED_COLUMNS = {
    "Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
}


def prepare_real_data(source: Path, output: Path, source_rows: int) -> pd.DataFrame:
    """Prepare an hourly classroom window from real minute measurements."""
    if not source.is_file():
        raise FileNotFoundError(
            f"Real power data not found at {source}. "
            "Follow the starter README and download the named Kaggle dataset."
        )
    raw = pd.read_csv(
        source, sep=";", nrows=source_rows, na_values=["?"], low_memory=False
    )
    missing = EXPECTED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    timestamp = pd.to_datetime(
        raw["Date"] + " " + raw["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    power = pd.to_numeric(raw["Global_active_power"], errors="coerce")
    minute = pd.DataFrame({"timestamp": timestamp, "global_active_power": power})
    minute = minute.dropna().sort_values("timestamp").set_index("timestamp")
    hourly = minute.resample("h").mean().dropna().reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output, index=False)
    return hourly


EXPECTED_DATA_ROWS = 2075259


def check_real_data(source: Path) -> int:
    """Verify the downloaded Kaggle file matches the README contract."""
    if not source.is_file():
        raise FileNotFoundError(
            f"Real power data not found at {source}. "
            "Follow the starter README and download the named Kaggle dataset."
        )
    first_ts: str | None = None
    last_ts: str | None = None
    row_count = 0
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip().split(";")
        for line in handle:
            row_count += 1
            fields = line.split(";", 2)
            first_ts = fields[0] + " " + fields[1] if first_ts is None else first_ts
            last_ts = fields[0] + " " + fields[1]
    missing = EXPECTED_COLUMNS - set(header)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if row_count != EXPECTED_DATA_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DATA_ROWS} data rows, found {row_count}"
        )
    if first_ts is None or last_ts is None:
        raise ValueError("Data file contains no data rows")
    start = pd.to_datetime(first_ts, format="%d/%m/%Y %H:%M:%S")
    end = pd.to_datetime(last_ts, format="%d/%m/%Y %H:%M:%S")
    print("REAL DATA CHECK PASSED")
    print(f"rows: {row_count}")
    print(f"time_range: {start} to {end}")
    print(f"columns: {sorted(EXPECTED_COLUMNS)}")
    return 0


def make_lagged(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose features use only earlier power measurements."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    lagged = pd.DataFrame({"timestamp": ordered["timestamp"].copy()})
    for lag in LAGS:
        # lag_N at row t holds the average power N hours before t.
        lagged[f"lag_{lag}"] = ordered["global_active_power"].shift(lag)
    # Calendar information known before the forecast hour, not future data.
    # hour_of_day captures the daily pattern; day_of_week captures weekly
    # routines (weekday vs weekend consumption), both known in advance.
    lagged["hour_of_day"] = ordered["timestamp"].dt.hour
    lagged["day_of_week"] = ordered["timestamp"].dt.dayofweek
    # Target: the average power of the hour right after the input window.
    lagged["target_next"] = ordered["global_active_power"].shift(-1)
    # Drop the warm-up rows (missing lags) and the last row (missing target).
    lagged = lagged.dropna().reset_index(drop=True)
    return lagged


def chronological_split(
    frame: pd.DataFrame, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = int(len(frame) * (1 - test_fraction))
    return frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()


def build_candidate() -> RandomForestRegressor:
    """Return one fixed practical regression model.

    A fixed seed keeps every run reproducible; the same test window is used
    for the persistence baseline and this model, so the comparison is fair.
    """
    # max_depth=15, min_samples_leaf=2: moderately less smoothing than the
    # starter defaults so the model can follow evening peaks better, which
    # raises high-demand recall while keeping MAE competitive.
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def forecast_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_kw": float(mean_absolute_error(actual, predicted)),
        "rmse_kw": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def warning_counts(
    actual: pd.Series, predicted: np.ndarray, threshold: float
) -> dict[str, int | float]:
    truth = actual.to_numpy() >= threshold
    flags = np.asarray(predicted) >= threshold
    tp = int(np.sum(truth & flags))
    fp = int(np.sum(~truth & flags))
    fn = int(np.sum(truth & ~flags))
    tn = int(np.sum(~truth & ~flags))
    return {
        "threshold_kw": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "recall_high_demand": tp / (tp + fn) if tp + fn else 0.0,
    }


def save_plot(
    timestamps: pd.Series,
    actual: pd.Series,
    baseline: np.ndarray,
    candidate: np.ndarray | None,
    output: Path,
) -> None:
    count = min(168, len(actual))
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(timestamps.iloc[:count], actual.iloc[:count], label="actual")
    axis.plot(timestamps.iloc[:count], baseline[:count], label="last-hour baseline")
    if candidate is not None:
        axis.plot(timestamps.iloc[:count], candidate[:count], label="random forest")
    axis.set_ylabel("Global active power (kW)")
    axis.set_title("First held-out week")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/hourly_power.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/household_power_consumption.txt"),
    )
    parser.add_argument("--source-rows", type=int, default=150000)
    args = parser.parse_args()
    if args.check_data:
        return check_real_data(args.source)
    if args.prepare:
        hourly = prepare_real_data(args.source, args.data, args.source_rows)
        print("REAL DATA PREPARATION PASSED")
        print(f"source_rows_requested: {args.source_rows}")
        print(f"hourly_rows: {len(hourly)}")
        print(f"start: {hourly['timestamp'].min()}")
        print(f"end: {hourly['timestamp'].max()}")
        print(f"output: {args.data}")
        return 0
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Prepared real data not found at {args.data}. "
            "Run python analyze.py --prepare first."
        )

    raw = pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values("timestamp")
    required = {"timestamp", "global_active_power"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    lagged = make_lagged(raw)
    train, test = chronological_split(lagged)
    features = [f"lag_{lag}" for lag in LAGS] + ["hour_of_day", "day_of_week"]
    baseline_prediction = test["lag_1"].to_numpy()
    high_demand_threshold = float(train["target_next"].quantile(0.90))
    result: dict[str, object] = {
        "dataset": "Household Electric Power Consumption",
        "source_rows": len(raw),
        "time_range": [str(raw.timestamp.min()), str(raw.timestamp.max())],
        "lags_hours": LAGS,
        "train_rows": len(train),
        "test_rows": len(test),
        "baseline": forecast_metrics(test["target_next"], baseline_prediction),
        "baseline_warnings": warning_counts(
            test["target_next"], baseline_prediction, high_demand_threshold
        ),
        "candidate": None,
    }
    candidate_prediction: np.ndarray | None = None
    try:
        model = build_candidate()
        model.fit(train[features], train["target_next"])
        candidate_prediction = model.predict(test[features])
        result["candidate"] = forecast_metrics(
            test["target_next"], candidate_prediction
        )
        result["candidate_warnings"] = warning_counts(
            test["target_next"], candidate_prediction, high_demand_threshold
        )
        errors = test[["timestamp", "target_next"]].copy()
        errors["predicted"] = candidate_prediction
        errors["absolute_error"] = (
            errors["target_next"] - errors["predicted"]
        ).abs()
        errors.nlargest(12, "absolute_error").to_csv(
            "largest_errors.csv", index=False
        )
    except NotImplementedError as error:
        result["candidate_todo"] = str(error)

    save_plot(
        test["timestamp"],
        test["target_next"],
        baseline_prediction,
        candidate_prediction,
        Path("forecast.png"),
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
