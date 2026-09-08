from __future__ import annotations

import bentoml
import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.preprocessing import preprocess_text

MODEL_PATH = "deployment/model"
MODEL_VERSION = "1"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

MODEL.to(DEVICE)
MODEL.eval()


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


@bentoml.service(
    name="arabic_sentiment_service"
)
class ArabicSentimentService:
    @bentoml.api
    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        cleaned_text = preprocess_text(
            request.text
        )

        inputs = TOKENIZER(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = MODEL(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=-1,
            )

        label = MODEL.config.id2label[
            int(predicted_index.item())
        ]

        return PredictionResponse(
            label=label,
            confidence=float(
                confidence.item()
            ),
            model_version=MODEL_VERSION,
        )