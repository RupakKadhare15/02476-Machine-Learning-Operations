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


""" from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from scipy.special import softmax

# Import your model class using the relative path inside your src structure
from src.toxic_comments.model import ToxicCommentsTransformer

# --- Configuration ---
CHECKPOINT_PATH = "models/bert-toxic-comments-classifier:v0/model.onnx"
# We need the base model name to load the correct tokenizer
BASE_MODEL_NAME = "vinai/bertweet-base" 

# Global variables to hold model and tokenizer
model = None
tokenizer = None

# --- Schemas ---
class ToxicCommentRequest(BaseModel):
    text: str

class ToxicCommentResponse(BaseModel):
    text: str
    label: str
    confidence: float
    is_toxic: bool

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        print("Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        print(f"Loading Model from {CHECKPOINT_PATH}...")
        # FORCE CPU LOAD to avoid MPS/device mismatch errors
        model = ToxicCommentsTransformer.load_from_checkpoint(
            CHECKPOINT_PATH, 
            map_location=torch.device("cpu")
        )
        model.eval()
        model.freeze()
        print("Model and Tokenizer loaded successfully!")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    finally:
        pass

app = FastAPI(title="Toxic Comment Classifier", lifespan=lifespan)

@app.get("/health")
def health_check():
    #Health check endpoint that reports service status and model readiness.
    return {"status": "running", "model_loaded": model is not None}

@app.post("/predict", response_model=ToxicCommentResponse)
def predict(request: ToxicCommentRequest):
   #Classifies input text as toxic or non-toxic and returns the prediction with confidence.
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model service not ready")

    try:
        # 1. Preprocessing (Tokenization)
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128 # Adjust based on your training config
        )

        # 2. Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
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
            is_toxic=(pred_idx == 1)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) """