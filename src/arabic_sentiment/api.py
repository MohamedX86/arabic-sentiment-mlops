from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from pydantic import BaseModel, Field

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Gauge,
)

from arabic_sentiment.preprocessing import preprocess_text


# =========================================================
# Model Configuration
# =========================================================

MODEL_PATH = "deployment/model"
MODEL_VERSION = "1"


MODEL = None
TOKENIZER = None


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(
    "arabic-sentiment-api"
)


# =========================================================
# Prometheus Metrics
# =========================================================

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)


REQUEST_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency milliseconds"
)


CONFIDENCE_SCORE = Histogram(
    "prediction_confidence_score",
    "Model confidence score"
)


CURRENT_PSI = Gauge(
    "prediction_psi",
    "Current PSI drift value"
)


# =========================================================
# Request / Response Models
# =========================================================

class PredictionRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="Arabic review text to classify."
    )


class PredictionResponse(BaseModel):

    label: str
    confidence: float
    model_version: str



# =========================================================
# Model Lifecycle
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL, TOKENIZER


    logger.info(
        "Loading local sentiment model..."
    )


    MODEL = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )


    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )


    MODEL.to(DEVICE)

    MODEL.eval()


    logger.info(
        "Model loaded | device=%s | version=%s",
        DEVICE,
        MODEL_VERSION
    )


    yield


    MODEL = None

    TOKENIZER = None


    logger.info(
        "Model resources released"
    )



# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(

    title="Arabic Sentiment API",

    description="AraBERT sentiment classification API",

    version="1.0.0",

    lifespan=lifespan
)
# =========================================================
# Request Monitoring Middleware
# =========================================================

@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next
):

    start_time = time.perf_counter()

    status_code = 500


    try:

        response = await call_next(request)

        status_code = response.status_code

        return response


    except Exception:

        logger.exception(
            "request_failed | path=%s",
            request.url.path
        )

        raise


    finally:

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000


        REQUEST_LATENCY.observe(
            latency_ms
        )


        logger.info(
            "request | method=%s | path=%s | status=%s | latency=%.2f ms",
            request.method,
            request.url.path,
            status_code,
            latency_ms
        )



# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Arabic Sentiment API is running."
    }



# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():

    if MODEL is None or TOKENIZER is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )


    return {

        "status": "healthy",

        "device": str(DEVICE),

        "model_version": MODEL_VERSION

    }



# =========================================================
# Prometheus Metrics Endpoint
# =========================================================

@app.get("/metrics")
def metrics():

    return Response(

        generate_latest(),

        media_type=CONTENT_TYPE_LATEST

    )



# =========================================================
# Prediction
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    if MODEL is None or TOKENIZER is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )


    try:

        REQUEST_COUNT.inc()


        cleaned_text = preprocess_text(
            request.text
        )


        inputs = TOKENIZER(

            cleaned_text,

            truncation=True,

            padding=True,

            max_length=128,

            return_tensors="pt"

        )


        inputs = {

            key: value.to(DEVICE)

            for key, value in inputs.items()

        }


        with torch.no_grad():

            outputs = MODEL(
                **inputs
            )


            probabilities = torch.softmax(

                outputs.logits,

                dim=-1

            )


            confidence, predicted_index = torch.max(

                probabilities,

                dim=-1

            )


        confidence_value = float(
            confidence.item()
        )


        CONFIDENCE_SCORE.observe(
            confidence_value
        )


        label = MODEL.config.id2label[

            int(predicted_index.item())

        ]


        result = PredictionResponse(

            label=label,

            confidence=confidence_value,

            model_version=MODEL_VERSION

        )


        logger.info(

            "prediction | label=%s | confidence=%.4f | version=%s",

            result.label,

            result.confidence,

            result.model_version

        )


        return result



    except Exception as exc:


        logger.exception(
            "prediction_failed"
        )


        raise HTTPException(

            status_code=500,

            detail=str(exc)

        ) from exc