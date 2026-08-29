from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.config import LABEL_NAMES, MODEL_NAME
from arabic_sentiment.preprocessing import preprocess_text


@dataclass
class PredictionResult:
    label: str
    confidence: float


class ArabicSentimentModel:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL_NAMES),
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        processed_text = preprocess_text(text)

        inputs: Dict[str, torch.Tensor] = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        confidence, predicted_index = torch.max(probabilities, dim=-1)

        label = LABEL_NAMES[predicted_index.item()]

        return PredictionResult(
            label=label,
            confidence=float(confidence.item()),
        )