from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.config import MODEL_NAME, RANDOM_SEED


TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")

MODEL_OUTPUT_DIR = Path("models/baseline_arabert")
REPORT_PATH = Path("reports/baseline_metrics.json")

MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 2

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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
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

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }


def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model.to(device)

    train_dataset = ArabicReviewDataset(
        train_df,
        tokenizer,
    )

    validation_dataset = ArabicReviewDataset(
        validation_df,
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):
            optimizer.zero_grad()

            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            outputs = model(
                **inputs,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                average_loss = total_loss / step

                print(
                    f"Step {step}/{len(train_loader)} "
                    f"- loss: {average_loss:.4f}"
                )

        metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )

        print(
            f"Validation accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Validation macro F1: "
            f"{metrics['f1_macro']:.4f}"
        )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        MODEL_OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        MODEL_OUTPUT_DIR
    )

    final_metrics = evaluate_model(
        model,
        validation_loader,
        device,
    )

    report = {
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "validation_accuracy": final_metrics[
            "accuracy"
        ],
        "validation_f1_macro": final_metrics[
            "f1_macro"
        ],
        "device": str(device),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTraining completed.")
    print(
        f"Model saved to: "
        f"{MODEL_OUTPUT_DIR}"
    )
    print(
        f"Metrics saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()