from fastapi.testclient import TestClient

from arabic_sentiment.api import app


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Arabic Sentiment API is running."
    }


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["device"] in {"cpu", "cuda"}


def test_positive_prediction() -> None:
    with TestClient(app) as client:
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


def test_negative_prediction() -> None:
    with TestClient(app) as client:
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


def test_neutral_prediction() -> None:
    with TestClient(app) as client:
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


def test_empty_text_validation() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": ""
            },
        )

    assert response.status_code == 422