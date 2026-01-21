import os
from contextlib import asynccontextmanager
from datetime import datetime as dt

import anyio
import pandas as pd
import torch
import torch.nn.functional as F
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
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


def add_to_db(
    pred_db_path: str,
    response: ToxicCommentResponse,
) -> None:
    """Simple function to add prediction to database."""
    now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(f'{pred_db_path}/predictions.csv'):
        with open(f'{pred_db_path}/predictions.csv', 'w') as file:
            file.write('time, comment_text, toxic\n')
    with open(f'{pred_db_path}/predictions.csv', 'a') as file:
        file.write(f'{now}, {response.text}, {response.label}\n')


app = FastAPI(title='Toxic Comment Classifier', lifespan=lifespan)


@app.get('/health')
def health_check():
    """Health check endpoint that reports service status and model readiness."""
    return {'status': 'running', 'model_loaded': model is not None}


@app.post('/predict', response_model=ToxicCommentResponse)
def predict(request: ToxicCommentRequest, pred_db_path: str = '/gcp/predictions_db'):
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

        response = ToxicCommentResponse(
            text=request.text,
            label=label_name,
            confidence=confidence,
            is_toxic=(pred_idx == 1),
        )
        add_to_db(pred_db_path, response)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/monitoring', response_class=HTMLResponse)
async def monitoring(
    data_dir: str = '/gcp/tweets-dataset/data',
    pred_db_path: str = '/gcp/predictions_db',
    n: int | None = None,
):
    """
    Simple get request method that returns a monitoring report.

    Args:
        data_dir: Directory containing the training data.
        pred_db_path: Directory containing the predictions database.
        n: Optional number of latest predictions to include in the report.

    """
    reference_data: pd.DataFrame = pd.read_csv(f'{data_dir}/train.csv')

    current_data = pd.read_csv(f'{pred_db_path}/predictions.csv')

    # Filter to last n rows if specified
    if n is not None and len(current_data) > n:
        current_data = current_data.sort_values('time').tail(n)

    current_data = current_data.drop(columns=['time'])

    data_drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = data_drift_report.run(current_data=current_data, reference_data=reference_data)
    snapshot.save_html('monitoring.html')

    async with await anyio.open_file('monitoring.html', encoding='utf-8') as f:
        html_content = f.read()

    os.remove('monitoring.html')

    return HTMLResponse(content=html_content, status_code=200)
