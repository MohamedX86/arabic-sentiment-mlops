from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models/baseline_arabert")

JSON_REPORT_PATH = Path("reports/baseline_test_metrics.json")
TEXT_REPORT_PATH = Path("reports/baseline_classification_report.txt")

MAX_LENGTH = 128
BATCH_SIZE = 8

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class ArabicReviewDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]

        encoding = self.tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            LABEL_TO_ID[row["label"]],
            dtype=torch.long,
        )

        return item


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    test_df = pd.read_csv(TEST_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)
    model.eval()

    test_dataset = ArabicReviewDataset(
        test_df,
        tokenizer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(**inputs)

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    f1_macro = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    target_names = [
        "negative",
        "neutral",
        "positive",
    ]

    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=target_names,
        digits=4,
    )

    report_dict = classification_report(
        all_labels,
        all_predictions,
        target_names=target_names,
        output_dict=True,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    JSON_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "test_accuracy": float(accuracy),
        "test_f1_macro": float(f1_macro),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "device": str(device),
    }

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        TEXT_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    print("\nTest evaluation completed.")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {f1_macro:.4f}")

    print("\nClassification Report:")
    print(report_text)

    print("Confusion Matrix:")
    print(matrix)

    print(
        f"\nJSON report saved to: {JSON_REPORT_PATH}"
    )
    print(
        f"Text report saved to: {TEXT_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()