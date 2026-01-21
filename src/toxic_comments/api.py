import tempfile
from contextlib import asynccontextmanager
from datetime import datetime as dt
from pathlib import Path

import anyio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# Import your model class using the relative path inside your src structure
from src.toxic_comments.model import ToxicCommentsTransformer

# --- Configuration ---
CHECKPOINT_PATH = 'lightning_logs/version_0/checkpoints/epoch=2-step=30.ckpt'
# We need the base model name to load the correct tokenizer
BASE_MODEL_NAME = 'vinai/bertweet-base'

# GCS bucket configuration
DATA_BUCKET_NAME = 'tweets-dataset'
PREDICTIONS_BUCKET_NAME = 'predictions_db'

# Global variables to hold model and tokenizer
model = None
tokenizer = None
gcs_client = None


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
    global model, tokenizer, gcs_client
    try:
        print('Initializing GCS client...')
        gcs_client = storage.Client()

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
    response: ToxicCommentResponse,
) -> None:
    """Simple function to add prediction to GCS bucket."""
    now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    bucket = gcs_client.bucket(PREDICTIONS_BUCKET_NAME)
    blob = bucket.blob('predictions.csv')

    # Download existing predictions or create new file
    try:
        existing_content = blob.download_as_text()
    except Exception:
        # File doesn't exist, create header
        existing_content = 'time, comment_text, toxic\n'

    # Append new prediction
    new_line = f'{now}, {response.text}, {response.label}\n'
    updated_content = existing_content + new_line

    # Upload back to GCS
    blob.upload_from_string(updated_content)


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

        response = ToxicCommentResponse(
            text=request.text,
            label=label_name,
            confidence=confidence,
            is_toxic=(pred_idx == 1),
        )
        add_to_db(response)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/monitoring', response_class=HTMLResponse)
async def monitoring(
    n: int | None = None,
):
    """
    Simple get request method that returns a monitoring report using embedding drift detection.

    Args:
        n: Optional number of latest predictions to include in the report.

    """
    try:
        # Create temporary directory for downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download files from GCS
            data_bucket = gcs_client.bucket(DATA_BUCKET_NAME)
            predictions_bucket = gcs_client.bucket(PREDICTIONS_BUCKET_NAME)

            # Download training embeddings
            embeddings_blob = data_bucket.blob('data/train_embeddings.csv')
            embeddings_path = temp_path / 'train_embeddings.csv'
            try:
                embeddings_blob.download_to_filename(str(embeddings_path))
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f'Training embeddings not found in GCS bucket {DATA_BUCKET_NAME}/data/train_embeddings.csv. '
                    'Run: python src/toxic_comments/data_drift.py --compute-embeddings',
                )

            reference_embeddings = pd.read_csv(embeddings_path)

            # Download predictions
            predictions_blob = predictions_bucket.blob('predictions.csv')
            predictions_path = temp_path / 'predictions.csv'
            try:
                predictions_blob.download_to_filename(str(predictions_path))
            except Exception:
                raise HTTPException(
                    status_code=404,
                    detail=f'Predictions file not found in GCS bucket {PREDICTIONS_BUCKET_NAME}/predictions.csv',
                )

            current_data = pd.read_csv(predictions_path)

            # Filter to last n rows if specified
            if n is not None and len(current_data) > n:
                current_data = current_data.sort_values('time').tail(n)

            # Download GLOVE model
            glove_blob = data_bucket.blob('data/glove.6B.50d.txt')
            glove_path = temp_path / 'glove.6B.50d.txt'
            try:
                glove_blob.download_to_filename(str(glove_path))
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f'GLOVE vectors not found in GCS bucket {DATA_BUCKET_NAME}/data/glove.6B.50d.txt. '
                    'Run: python src/toxic_comments/data_drift.py --compute-embeddings',
                )

            glove_model = _load_glove_model(glove_path)

            # Compute embeddings for current predictions
            current_embeddings = _get_sentence_embeddings(current_data['comment_text'], glove_model)
            current_embeddings_df = pd.DataFrame(current_embeddings)
            current_embeddings_df.columns = [f'emb_{i}' for i in range(current_embeddings_df.shape[1])]

            # Ensure both datasets have the same columns
            reference_columns = reference_embeddings.columns.tolist()
            current_embeddings_df = current_embeddings_df[reference_columns]

            # Calculate cosine similarity between reference and current embeddings
            ref_mean = reference_embeddings.mean(axis=0).values.reshape(1, -1)
            curr_mean = current_embeddings_df.mean(axis=0).values.reshape(1, -1)
            similarity = cosine_similarity(ref_mean, curr_mean)[0][0]

            # Create drift report using DataDriftPreset on embedding columns
            report = Report(metrics=[DataDriftPreset()])

            # Run report on subset of data for performance
            ref_sample = reference_embeddings.head(500)
            curr_sample = current_embeddings_df.head(min(500, len(current_embeddings_df)))

            snapshot = report.run(reference_data=ref_sample, current_data=curr_sample)

            # Save and return HTML
            report_path = temp_path / 'monitoring.html'
            snapshot.save_html(str(report_path))

            async with await anyio.open_file(str(report_path), encoding='utf-8') as f:
                html_content = await f.read()

            # Add cosine similarity info to HTML
            similarity_info = f'<p style="padding:10px;background-color:#f0f0f0;">Cosine Similarity between reference and current embeddings: {similarity:.4f}</p>'
            html_content = html_content.replace('<body>', f'<body>{similarity_info}')

            return HTMLResponse(content=html_content, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating monitoring report: {str(e)}')


def _load_glove_model(glove_file: Path):
    """
    Load GLOVE vectors from a text file.

    Args:
        glove_file: Path to the GLOVE vectors file

    Returns:
        Dictionary mapping words to their vector representations

    """
    glove_model = {}

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding

    return glove_model


def _get_sentence_embeddings(texts, glove_model, dim=50):
    """
    Convert texts to sentence embeddings by averaging word vectors.

    Args:
        texts: Series or list of text strings
        glove_model: Dictionary of word to vector mappings
        dim: Dimensionality of the embeddings

    Returns:
        numpy array of sentence embeddings

    """
    embeddings = []

    for text in texts:
        # Tokenize by splitting on spaces (simple tokenization)
        words = str(text).lower().split()

        # Average word vectors
        word_vecs = []
        for word in words:
            if word in glove_model:
                word_vecs.append(glove_model[word])

        # If no words found, use zero vector
        if len(word_vecs) == 0:
            sentence_vec = np.zeros(dim)
        else:
            sentence_vec = np.mean(word_vecs, axis=0)

        embeddings.append(sentence_vec)

    return np.array(embeddings)
