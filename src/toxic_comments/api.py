from contextlib import asynccontextmanager
import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from scipy.special import softmax
from google.cloud import storage

ONNX_MODEL_PATH = "models/bert-toxic-comments-classifier:v0/model.onnx"
BASE_MODEL_NAME = 'vinai/bertweet-base'

# Global variables to hold model and tokenizer
model = None
tokenizer = None

ID_LABEL = {0: 'NON-TOXIC', 1: 'TOXIC'}

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
            blob = bucket.blob("model.onnx")  # The name of the file inside the bucket
            blob.download_to_filename(ONNX_MODEL_PATH)
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

@app.post('/predict', response_model=ToxicCommentResponse)
def predict(request: ToxicCommentRequest):
    """Classifies input text using ONNX runtime."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail='Model service not ready')

    try:
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

        # Check if logits have the expected shape
        if logits.shape[1] != 2:
            raise ValueError(f'Unexpected logits shape: {logits.shape}')

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

    except Exception as e:
        # Catch errors specifically related to shape or ONNX runtime
        raise HTTPException(status_code=500, detail=str(e))