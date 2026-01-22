from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.toxic_comments.api as api


@asynccontextmanager
async def noop_lifespan(app):  # noqa: ARG001
    """No-op lifespan context manager."""
    yield


@pytest.fixture()
def client():
    """TestClient fixture with mocked model/tokenizer loading."""
    # Prevent real tokenizer/model loading during TestClient startup
    api.app.router.lifespan_context = noop_lifespan
    return TestClient(api.app, raise_server_exceptions=False)


# Dummy classes to mock tokenizer and model behavior
class DummyTokenizer:
    """A dummy tokenizer that can optionally raise an error."""

    def __init__(self, should_raise: bool = False):
        """Initialize the dummy tokenizer."""
        self.should_raise = should_raise

    def __call__(self, text: str, **kwargs):
        """Tokenize the input text or raise an error."""
        if self.should_raise:
            raise RuntimeError('boom')

        return {
            'input_ids': np.array([[1, 2, 3]]),
            'attention_mask': np.array([[1, 1, 1]]),
        }


class DummyModel:
    """A dummy model that returns predefined logits or raises an error."""

    def __init__(
        self,
        logits: np.ndarray,
        should_raise: bool = False,
    ):
        """Initialize the dummy model."""
        self._logits = logits
        self.should_raise = should_raise

    def eval(self):
        """Dummy eval method."""
        return self

    def freeze(self):
        """Dummy freeze method."""
        return self

    def run(self, output_names=None, ort_inputs=None):
        """Return logits or raise an error."""
        if self.should_raise:
            raise RuntimeError("boom")
        # must be indexable because your code does ...[0]
        return [self._logits]


# first test: health endpoint
def test_health_returns_200_and_shape(client):
    """Test /health endpoint returns 200 and expected keys."""
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert 'status' in body
    assert 'model_loaded' in body


def test_health_reports_model_not_loaded(client):
    """Test /health reports model_loaded as False when model is None."""
    with patch('src.toxic_comments.api.model', None):
        response = client.get('/health')
        assert response.json() == {'status': 'running', 'model_loaded': False}


def test_health_reports_model_loaded(client):
    """Test /health reports model_loaded as True when model is present."""
    with patch('src.toxic_comments.api.model', object()):
        response = client.get('/health')
        assert response.json() == {'status': 'running', 'model_loaded': True}


# /predict readiness (503)
@pytest.mark.parametrize(
    'model_value, tokenizer_value',
    [
        (None, None),
        (None, object()),
        (object(), None),
    ],
)
def test_predict_returns_503_when_not_ready(client, model_value, tokenizer_value):
    """Test /predict returns 503 if model or tokenizer is not loaded."""
    with patch('src.toxic_comments.api.model', model_value), patch('src.toxic_comments.api.tokenizer', tokenizer_value):
        response = client.post('/predict', json={'text': 'hello'})
        assert response.status_code == 503
        assert response.json()['detail'] == 'Model service not ready'


# /predict successful prediction
def test_predict_non_toxic_success(client):
    """Test /predict returns correct response for non-toxic input."""
    logits = np.array([[10.0, 0.0]])

    with (
        patch('src.toxic_comments.api.tokenizer', DummyTokenizer()),
        patch('src.toxic_comments.api.model', DummyModel(logits)),
        patch('src.toxic_comments.api.add_to_db'),
    ):
        response = client.post('/predict', json={'text': 'hello'})
        body = response.json()

        assert response.status_code == 200
        assert body['label'] == 'NON-TOXIC'
        assert body['is_toxic'] is False
        assert isinstance(body['confidence'], float)
        assert 0.0 <= body['confidence'] <= 1.0


def test_predict_toxic_success(client):
    """Test /predict returns correct response for toxic input."""
    logits = np.array([[0.0, 10.0]])

    with (
        patch('src.toxic_comments.api.tokenizer', DummyTokenizer()),
        patch('src.toxic_comments.api.model', DummyModel(logits)),
        patch('src.toxic_comments.api.add_to_db'),
    ):
        response = client.post('/predict', json={'text': 'you are awful'})
        body = response.json()

        print(f"The problem is , {body}")

        assert response.status_code == 200
        assert body['label'] == 'TOXIC'
        assert body['is_toxic'] is True
        assert body['confidence'] >= 0.5


# /predict error handling
def test_predict_500_if_tokenizer_raises(client):
    """Test /predict returns 500 if tokenizer raises an error."""
    with (
        patch('src.toxic_comments.api.tokenizer', DummyTokenizer(should_raise=True)),
        patch('src.toxic_comments.api.model', DummyModel(np.array([[0.0, 1.0]]))),
    ):
        response = client.post('/predict', json={'text': 'hi'})
        assert response.status_code == 500
        assert 'boom' in response.json()['detail']


def test_predict_500_if_model_raises(client):
    """Test /predict returns 500 if model raises an error."""
    with (
        patch('src.toxic_comments.api.tokenizer', DummyTokenizer()),
        patch(
            'src.toxic_comments.api.model',
            DummyModel(np.array([[0.0, 1.0]]), should_raise=True),
        ),
    ):
        response = client.post('/predict', json={'text': 'hi'})
        assert response.status_code == 500
        assert 'boom' in response.json()['detail']


def test_predict_500_invalid_logits_shape(client):
    """Test /predict returns 500 if logits have invalid shape."""
    with (
        patch('src.toxic_comments.api.tokenizer', DummyTokenizer()),
        patch('src.toxic_comments.api.model', DummyModel(np.array([[1.0]]))),
    ):
        response = client.post('/predict', json={'text': 'hi'})
        assert response.status_code == 500

