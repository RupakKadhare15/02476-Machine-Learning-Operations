from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from unittest.mock import patch
import torch
from fastapi.testclient import TestClient
import src.toxic_comments.api as api

@asynccontextmanager
async def noop_lifespan(app):  # noqa: ARG001
    yield


@pytest.fixture()
def client():
    # Prevent real tokenizer/model loading during TestClient startup
    api.app.router.lifespan_context = noop_lifespan
    return TestClient(api.app, raise_server_exceptions=False)

# Dummy classes to mock tokenizer and model behavior
class DummyTokenizer:
    def __init__(self, should_raise: bool = False):
        self.should_raise = should_raise

    def __call__(self, text: str, **kwargs):
        if self.should_raise:
            raise RuntimeError("boom")

        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class DummyModel:
    def __init__(
        self,
        logits: torch.Tensor,
        id2label: Optional[Dict[int, str]] = None,
        should_raise: bool = False,
    ):
        self.logits = logits
        self.should_raise = should_raise
        self.config = SimpleNamespace(id2label=id2label or {0: "non-toxic", 1: "toxic"})

    def eval(self):
        return self

    def freeze(self):
        return self

    def __call__(self, **inputs):
        if self.should_raise:
            raise RuntimeError("boom")
        return SimpleNamespace(logits=self.logits)
    

# first test: health endpoint
def test_health_returns_200_and_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body


def test_health_reports_model_not_loaded(client):
    with patch("src.toxic_comments.api.model", None):
        response = client.get("/health")
        assert response.json() == {"status": "running", "model_loaded": False}


def test_health_reports_model_loaded(client):
    with patch("src.toxic_comments.api.model", object()):
        response = client.get("/health")
        assert response.json() == {"status": "running", "model_loaded": True}

# /predict readiness (503)
@pytest.mark.parametrize(
    "model_value, tokenizer_value",
    [
        (None, None),
        (None, object()),
        (object(), None),
    ],
)
def test_predict_returns_503_when_not_ready(client, model_value, tokenizer_value):
    with patch("src.toxic_comments.api.model", model_value), patch(
        "src.toxic_comments.api.tokenizer", tokenizer_value
    ):
        response = client.post("/predict", json={"text": "hello"})
        assert response.status_code == 503
        assert response.json()["detail"] == "Model service not ready"

# /predict successful prediction
def test_predict_non_toxic_success(client):
    logits = torch.tensor([[10.0, 0.0]])

    with patch("src.toxic_comments.api.tokenizer", DummyTokenizer()), patch(
        "src.toxic_comments.api.model", DummyModel(logits)
    ):
        response = client.post("/predict", json={"text": "hello"})
        body = response.json()

        assert response.status_code == 200
        assert body["label"] == "non-toxic"
        assert body["is_toxic"] is False
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0


def test_predict_toxic_success(client):
    logits = torch.tensor([[0.0, 10.0]])

    with patch("src.toxic_comments.api.tokenizer", DummyTokenizer()), patch(
        "src.toxic_comments.api.model", DummyModel(logits)
    ):
        response = client.post("/predict", json={"text": "you are awful"})
        body = response.json()

        assert response.status_code == 200
        assert body["label"] == "toxic"
        assert body["is_toxic"] is True
        assert body["confidence"] >= 0.5