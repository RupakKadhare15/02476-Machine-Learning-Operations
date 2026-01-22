from contextlib import asynccontextmanager
import os
import time

import psutil
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from transformers import AutoTokenizer
from scipy.special import softmax
from google.cloud import storage

ONNX_MODEL_PATH = "models/model.onnx"
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

ID_LABEL = {0:'NON-TOXIC', 1: 'TOXIC'}

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
    """Handles automatic model downloading (if missing) and loading on startup."""
    global model, tokenizer
    try:
        if not os.path.exists(ONNX_MODEL_PATH):
            print(f"Model not found at {ONNX_MODEL_PATH}. Downloading from GCS...")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
            
            # Download from GCS
            client = storage.Client()
            bucket = client.bucket("models_bertoxic")
            blob = bucket.blob("model.onnx")  
            blob.download_to_filename(ONNX_MODEL_PATH)

            blob_data = bucket.blob("model.onnx.data")
            data_path = ONNX_MODEL_PATH + ".data"
            blob_data.download_to_filename(data_path)
            
            print("Download complete.")
        print('Loading Tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        print(f'Loading ONNX Model from {ONNX_MODEL_PATH}...')

        # Load the ONNX model
        model = ort.InferenceSession(ONNX_MODEL_PATH)
        
        print('ONNX Model and Tokenizer loaded successfully!')
        yield
    except Exception as e:
        print(f'Error loading model: {e}')
        raise e
    finally:
        pass

app = FastAPI(title='Toxic Comment Classifier (ONNX)', lifespan=lifespan)

@app.get('/health')
def health_check():
    """Health check endpoint that reports service status and model readiness."""
    return {'status': 'running', 'model_loaded': model is not None}

@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/predict', response_model=ToxicCommentResponse)
def predict(request: ToxicCommentRequest):
    """Classifies input text using ONNX runtime."""
    endpoint = "/predict"
    method = "POST"
    
    if not model or not tokenizer:
        # record as 503 too
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status="503").inc()
        raise HTTPException(status_code=503, detail='Model service not ready')

    start = time.perf_counter()
    status = "200"

    try:
        update_system_metrics()

        # 1. Preprocessing (Tokenization)
        inputs = tokenizer(
            request.text,
            return_tensors='np', 
            padding="max_length", 
            truncation=True,
            max_length=128,
        )

        # 2. Inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        logits = model.run(None, ort_inputs)[0]

        if logits.shape[1] != 2:
            raise ValueError(f'Unexpected logits shape: {logits.shape}. Expected 2 labels.')

        # 3. Postprocessing
        probs = softmax(logits, axis=1)

        # Get the predicted class index (0 or 1)
        pred_idx = np.argmax(probs, axis=1)[0]
        confidence = float(probs[0][pred_idx])

        # Retrieve label name
        label_name = ID_LABEL[pred_idx]

        return ToxicCommentResponse(
            text=request.text,
            label=label_name,
            confidence=confidence,
            is_toxic=bool(pred_idx == 1),
        )

    except HTTPException:
        # if you ever raise HTTPException inside, preserve its status in metrics
        status = "500"
        raise
    except Exception as e:
        status = "500"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = time.perf_counter() - start
        HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint, method=method).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status).inc()
