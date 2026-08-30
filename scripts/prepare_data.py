from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from arabic_sentiment.preprocessing import preprocess_text


RAW_DATA_PATH = Path("data/raw/PROD.csv")
PROCESSED_DIR = Path("data/processed")

LABEL_MAPPING = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}

RANDOM_SEED = 42


def load_data() -> pd.DataFrame:
    """Load the raw Arabic product review dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean review text and map sentiment labels."""
    cleaned = df.copy()

    cleaned["text"] = cleaned["text"].apply(preprocess_text)
    cleaned["label"] = cleaned["polarity"].map(LABEL_MAPPING)

    cleaned = cleaned[["text", "label"]]

    return cleaned


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits."""

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    return train_df, val_df, test_df


def save_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save processed splits to disk."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


def main() -> None:
    """Run the complete data preparation pipeline."""

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)

    train_df, val_df, test_df = split_data(cleaned_df)

    save_data(train_df, val_df, test_df)

    print("Data preparation completed successfully.")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()