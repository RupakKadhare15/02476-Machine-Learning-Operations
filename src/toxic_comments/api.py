from contextlib import asynccontextmanager
import os
import time

import psutil
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from transformers import AutoTokenizer

# Import your model class using the relative path inside your src structure
from src.toxic_comments.model import ToxicCommentsTransformer

# --- Configuration ---
CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=2-step=30.ckpt'
# We need the base model name to load the correct tokenizer
BASE_MODEL_NAME = 'vinai/bertweet-base'

# Global variables to hold model and tokenizer
model = None
tokenizer = None

# --- Prometheus Metrics ---
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "method"],
    # simple buckets that work fine for inference APIs
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

PROCESS_RESIDENT_MEMORY_BYTES = Gauge(
    "process_resident_memory_bytes",
    "Resident set size (RSS) memory in bytes",
)

PROCESS_CPU_PERCENT = Gauge(
    "process_cpu_percent",
    "Process CPU utilization percentage",
)

# Process handle for system metrics
_proc = psutil.Process(os.getpid())


def update_system_metrics() -> None:
    """Update a couple of simple process/system metrics."""
    # RSS memory in bytes
    PROCESS_RESIDENT_MEMORY_BYTES.set(_proc.memory_info().rss)
    # CPU percent since last call (psutil needs to be "primed" once)
    PROCESS_CPU_PERCENT.set(_proc.cpu_percent(interval=None))

# --- Schemas ---
class ToxicCommentRequest(BaseModel):
    """Request schema for toxic comment classification."""

    text: str


class ToxicCommentResponse(BaseModel):
    """Response schema for toxic comment classification."""

    text: str
    label: str
    confidence: float
    is_toxic: bool


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that handles model loading and cleanup."""
    global model, tokenizer
    try:
        print('Loading Tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        print(f'Loading Model from {CHECKPOINT_PATH}...')
        # FORCE CPU LOAD to avoid MPS/device mismatch errors
        model = ToxicCommentsTransformer.load_from_checkpoint(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        model.eval()
        model.freeze()
        print('Model and Tokenizer loaded successfully!')
        yield
    except Exception as e:
        print(f'Error loading model: {e}')
        raise e
    finally:
        pass


app = FastAPI(title='Toxic Comment Classifier', lifespan=lifespan)


@app.get('/health')
def health_check():
    """Health check endpoint that reports service status and model readiness."""
    return {'status': 'running', 'model_loaded': model is not None}


@app.post('/predict', response_model=ToxicCommentResponse)
def predict(request: ToxicCommentRequest):
    """Classifies input text as toxic or non-toxic and returns the prediction with confidence."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail='Model service not ready')

    try:
        # 1. Preprocessing (Tokenization)
        inputs = tokenizer(
            request.text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,  # Adjust based on your training config
        )

        # 2. Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Check if logits have the expected shape
        if logits.shape[1] != 2:
            raise ValueError(f'Unexpected logits shape: {logits.shape}')

        # 3. Postprocessing
        probs = F.softmax(logits, dim=1)

        # Get the predicted class index (0 or 1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

        # Retrieve label name from the model config
        label_name = model.config.id2label[pred_idx]

        return ToxicCommentResponse(
            text=request.text,
            label=label_name,
            confidence=confidence,
            is_toxic=(pred_idx == 1),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
