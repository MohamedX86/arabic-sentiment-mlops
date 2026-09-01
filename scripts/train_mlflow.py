from __future__ import annotations

import argparse

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.config import MODEL_NAME, RANDOM_SEED


TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

EXPERIMENT_NAME = "arabic-sentiment-arabert"

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
        tokenizer,
        max_length: int,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AraBERT with MLflow tracking."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--weighted-loss",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(
    train_df: pd.DataFrame,
    device: torch.device,
) -> torch.Tensor:
    labels = train_df["label"].map(LABEL_TO_ID).to_numpy()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=labels,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def evaluate_model(
    model,
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
    args = parse_args()

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
        args.max_length,
    )

    validation_dataset = ArabicReviewDataset(
        validation_df,
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    if args.weighted_loss:
        class_weights = calculate_class_weights(
            train_df,
            device,
        )

        loss_function = CrossEntropyLoss(
            weight=class_weights
        )

        print(
            f"Class weights: "
            f"{class_weights.detach().cpu().tolist()}"
        )
    else:
        loss_function = CrossEntropyLoss()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=args.run_name
    ) as run:
        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_param(
            "model_name",
            MODEL_NAME,
        )

        mlflow.log_param(
            "learning_rate",
            args.learning_rate,
        )

        mlflow.log_param(
            "batch_size",
            args.batch_size,
        )

        mlflow.log_param(
            "epochs",
            args.epochs,
        )

        mlflow.log_param(
            "max_length",
            args.max_length,
        )

        mlflow.log_param(
            "weighted_loss",
            args.weighted_loss,
        )

        mlflow.log_param(
            "random_seed",
            RANDOM_SEED,
        )

        best_f1 = -1.0

        for epoch in range(args.epochs):
            model.train()

            total_loss = 0.0

            print(
                f"\nEpoch {epoch + 1}/{args.epochs}"
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

                outputs = model(**inputs)

                loss = loss_function(
                    outputs.logits,
                    labels,
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

                if step % 50 == 0:
                    average_loss = total_loss / step

                    print(
                        f"Step {step}/{len(train_loader)} "
                        f"- loss: {average_loss:.4f}"
                    )

            average_train_loss = (
                total_loss / len(train_loader)
            )

            metrics = evaluate_model(
                model,
                validation_loader,
                device,
            )

            mlflow.log_metric(
                "train_loss",
                average_train_loss,
                step=epoch + 1,
            )

            mlflow.log_metric(
                "accuracy",
                metrics["accuracy"],
                step=epoch + 1,
            )

            mlflow.log_metric(
                "f1_macro",
                metrics["f1_macro"],
                step=epoch + 1,
            )

            print(
                f"Validation accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Validation macro F1: "
                f"{metrics['f1_macro']:.4f}"
            )

            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]

        final_metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )

        mlflow.log_metric(
            "final_accuracy",
            final_metrics["accuracy"],
        )

        mlflow.log_metric(
            "final_f1_macro",
            final_metrics["f1_macro"],
        )

        print("\nLogging model to MLflow...")

        model_info = mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer,
            },
            task="text-classification",
            name="model",
        )

        mlflow.set_tag(
            "task",
            "arabic_sentiment_classification",
        )

        mlflow.set_tag(
            "device",
            str(device),
        )

        mlflow.set_tag(
            "dataset",
            "PROD",
        )

        mlflow.set_tag(
            "model_logged",
            "true",
        )

        print("\nRun completed.")
        print(
            f"Final validation accuracy: "
            f"{final_metrics['accuracy']:.4f}"
        )
        print(
            f"Final validation macro F1: "
            f"{final_metrics['f1_macro']:.4f}"
        )
        print(
            f"MLflow run ID: "
            f"{run.info.run_id}"
        )
        print(
            f"MLflow model URI: "
            f"{model_info.model_uri}"
        )


if __name__ == "__main__":
    main()