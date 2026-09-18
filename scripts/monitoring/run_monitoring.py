import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT / "data" / "processed" / "train.csv"
PROD_PATH = ROOT / "data" / "raw" / "PROD.csv"

OUTPUT_DIR = ROOT / "reports" / "monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PSI_THRESHOLD = 0.25


# ============================================================
# HELPERS
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    return df


def find_target_column(
    df: pd.DataFrame,
    candidates,
) -> str:
    """Find the first available target column."""

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"None of the expected columns {candidates} "
        f"were found. Available columns: {list(df.columns)}"
    )


def prepare_monitoring_data(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Create monitoring features."""

    if "text" not in df.columns:
        raise ValueError(
            "'text' column is required. "
            f"Available columns: {list(df.columns)}"
        )

    result = pd.DataFrame()

    # --------------------------------------------------------
    # Text length
    # --------------------------------------------------------

    result["text_length"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    result["sentiment"] = (
        df[target_column]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return result


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Interpretation:

    PSI <= 0.10
        Low change

    0.10 < PSI <= 0.25
        Moderate change

    PSI > 0.25
        Significant drift
    """

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if reference.empty or current.empty:
        return 0.0

    # --------------------------------------------------------
    # Build bins using reference distribution
    # --------------------------------------------------------

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = np.unique(
        reference.quantile(
            quantiles
        ).values
    )

    # Not enough unique values
    if len(edges) < 3:
        return 0.0

    # Make sure all values can be included
    edges[0] = -np.inf
    edges[-1] = np.inf

    # --------------------------------------------------------
    # Reference distribution
    # --------------------------------------------------------

    reference_counts = pd.cut(
        reference,
        bins=edges,
        include_lowest=True,
    ).value_counts(
        sort=False
    )

    # --------------------------------------------------------
    # Current distribution
    # --------------------------------------------------------

    current_counts = pd.cut(
        current,
        bins=edges,
        include_lowest=True,
    ).value_counts(
        sort=False
    )

    reference_pct = (
        reference_counts / len(reference)
    )

    current_pct = (
        current_counts / len(current)
    )

    # --------------------------------------------------------
    # Avoid zero division
    # --------------------------------------------------------

    epsilon = 1e-6

    reference_pct = reference_pct.clip(
        lower=epsilon
    )

    current_pct = current_pct.clip(
        lower=epsilon
    )

    # --------------------------------------------------------
    # PSI formula
    # --------------------------------------------------------

    psi = np.sum(
        (
            current_pct - reference_pct
        )
        *
        np.log(
            current_pct / reference_pct
        )
    )

    return float(psi)


def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    """Calculate PSI for categorical distributions."""

    reference = (
        reference
        .fillna("missing")
        .astype(str)
    )

    current = (
        current
        .fillna("missing")
        .astype(str)
    )

    # --------------------------------------------------------
    # Combine all categories
    # --------------------------------------------------------

    categories = sorted(
        set(reference.unique()).union(
            set(current.unique())
        )
    )

    if not categories:
        return 0.0

    reference_counts = (
        reference.value_counts()
    )

    current_counts = (
        current.value_counts()
    )

    # --------------------------------------------------------
    # Reference percentages
    # --------------------------------------------------------

    reference_pct = np.array(
        [
            reference_counts.get(
                category,
                0,
            )
            / len(reference)
            for category in categories
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Current percentages
    # --------------------------------------------------------

    current_pct = np.array(
        [
            current_counts.get(
                category,
                0,
            )
            / len(current)
            for category in categories
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Avoid zero values
    # --------------------------------------------------------

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None,
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None,
    )

    # --------------------------------------------------------
    # PSI formula
    # --------------------------------------------------------

    psi = np.sum(
        (
            current_pct - reference_pct
        )
        *
        np.log(
            current_pct / reference_pct
        )
    )

    return float(psi)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Arabic Sentiment - Monitoring")
    print("=" * 60)

    # ========================================================
    # Load datasets
    # ========================================================

    print("\nLoading datasets...")

    reference_raw = load_dataset(
        REFERENCE_PATH
    )

    production_raw = load_dataset(
        PROD_PATH
    )

    print(
        f"Reference rows : {len(reference_raw)}"
    )

    print(
        f"Production rows: {len(production_raw)}"
    )

    # ========================================================
    # Find target columns
    # ========================================================

    reference_target = find_target_column(
        reference_raw,
        [
            "label",
            "polarity",
        ],
    )

    production_target = find_target_column(
        production_raw,
        [
            "polarity",
            "label",
        ],
    )

    print(
        f"\nReference target column : "
        f"{reference_target}"
    )

    print(
        f"Production target column: "
        f"{production_target}"
    )

    # ========================================================
    # Prepare monitoring datasets
    # ========================================================

    reference = prepare_monitoring_data(
        reference_raw,
        reference_target,
    )

    production = prepare_monitoring_data(
        production_raw,
        production_target,
    )

    # ========================================================
    # Calculate PSI
    # ========================================================

    print("\nCalculating PSI...")

    text_length_psi = calculate_psi(
        reference["text_length"],
        production["text_length"],
    )

    sentiment_psi = calculate_categorical_psi(
        reference["sentiment"],
        production["sentiment"],
    )

    # ========================================================
    # Detect drift
    # ========================================================

    drift_detected = (
        text_length_psi > PSI_THRESHOLD
        or
        sentiment_psi > PSI_THRESHOLD
    )

    # ========================================================
    # Metrics object
    # ========================================================

    metrics = {
        "reference_rows": len(reference),
        "production_rows": len(production),
        "reference_target": (
            reference_target
        ),
        "production_target": (
            production_target
        ),
        "text_length_psi": round(
            text_length_psi,
            6,
        ),
        "sentiment_psi": round(
            sentiment_psi,
            6,
        ),
        "psi_threshold": PSI_THRESHOLD,
        "drift_detected": bool(
            drift_detected
        ),
    }

    # ========================================================
    # Save monitoring metrics
    # ========================================================

    metrics_path = (
        OUTPUT_DIR
        / "monitoring_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nMetrics saved to:"
        f"\n{metrics_path}"
    )

    # ========================================================
    # Evidently Data Drift Report
    # ========================================================

    print(
        "\nGenerating Evidently report..."
    )

    evidently_reference = (
        reference.copy()
    )

    evidently_current = (
        production.copy()
    )

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    # Evidently 0.7.x returns a Snapshot
    snapshot = report.run(
        reference_data=evidently_reference,
        current_data=evidently_current,
    )

    evidently_path = (
        OUTPUT_DIR
        / "evidently_data_drift.html"
    )

    # Save HTML from the returned snapshot
    snapshot.save_html(
        str(evidently_path)
    )

    print(
        f"Evidently report saved to:"
        f"\n{evidently_path}"
    )

    # ========================================================
    # Human-readable status
    # ========================================================

    status = (
        "DRIFT DETECTED"
        if drift_detected
        else
        "NO SIGNIFICANT DRIFT"
    )

    text_length_status = (
        "DRIFT"
        if text_length_psi > PSI_THRESHOLD
        else
        "OK"
    )

    sentiment_status = (
        "DRIFT"
        if sentiment_psi > PSI_THRESHOLD
        else
        "OK"
    )

    print("\nMonitoring Summary:")

    print("\nGenerated files:")
    print(f"- {metrics_path}")
    print(f"- {evidently_path}")

    print("\nMonitoring completed successfully.")


if __name__ == "__main__":
    main()