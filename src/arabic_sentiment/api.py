from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.preprocessing import preprocess_text

MODEL_PATH = "deployment/model"

MODEL = None
TOKENIZER = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Arabic review text to classify.",
    )


class PredictionResponse(BaseModel):
    label: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER

    print("Loading local sentiment model...")

    MODEL = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Model loaded on device: {DEVICE}")

    yield

    MODEL = None
    TOKENIZER = None


app = FastAPI(
    title="Arabic Sentiment API",
    description="AraBERT sentiment classification API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Arabic Sentiment API is running."
    }


@app.get("/health")
def health() -> dict[str, str]:
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "status": "healthy",
        "device": str(DEVICE),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
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
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc