from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
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

# -------------------------
# Dummy tokenizer & model
# -------------------------
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