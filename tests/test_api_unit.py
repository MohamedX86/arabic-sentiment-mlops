from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

import arabic_sentiment.api as api


class FakeTokenizer:
    def __call__(
        self,
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    ):
        if "سيئ" in text or "لن أشتريه" in text:
            class_id = 0

        elif "استلمت" in text or "الموعد" in text:
            class_id = 1

        else:
            class_id = 2

        return {
            "input_ids": torch.tensor([[class_id]]),
            "attention_mask": torch.tensor([[1]]),
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            id2label={
                0: "negative",
                1: "neutral",
                2: "positive",
            }
        )

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        class_id = int(input_ids[0][0].item())

        logits = torch.full(
            (1, 3),
            -5.0,
        )

        logits[0, class_id] = 5.0

        return SimpleNamespace(
            logits=logits
        )


@pytest.fixture(autouse=True)
def mock_model_loading(monkeypatch):
    monkeypatch.setattr(
        api.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    monkeypatch.setattr(
        api.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )


def test_root_endpoint():
    with TestClient(api.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Arabic Sentiment API is running."
    }


def test_health_endpoint():
    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["device"] in {
        "cpu",
        "cuda",
    }


def test_positive_prediction():
    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "المنتج ممتاز جدا وأنصح به"
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["label"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0


def test_negative_prediction():
    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "المنتج سيئ جدا ولن أشتريه مرة أخرى"
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["label"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0


def test_neutral_prediction():
    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "استلمت المنتج اليوم ووصل في الموعد المحدد"
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["label"] == "neutral"
    assert 0.0 <= body["confidence"] <= 1.0


def test_empty_text_validation():
    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "text": ""
            },
        )

    assert response.status_code == 422