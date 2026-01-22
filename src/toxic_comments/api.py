import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime as dt
from pathlib import Path

import anyio
import numpy as np
import onnxruntime as ort
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage
from pydantic import BaseModel
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

ONNX_MODEL_PATH = "models/model.onnx"
BASE_MODEL_NAME = 'vinai/bertweet-base'

# GCS bucket configuration
DATA_BUCKET_NAME = 'tweets-dataset'
PREDICTIONS_BUCKET_NAME = 'predictions_db'

# Global variables to hold model and tokenizer
model = None
tokenizer = None
gcs_client = None

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
    global model, tokenizer, gcs_client
    try:
        gcs_client = storage.Client()
        if not os.path.exists(ONNX_MODEL_PATH):
            print(f"Model not found at {ONNX_MODEL_PATH}. Downloading from GCS...")

            # Ensure directory exists
            os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)

            # Download from GCS
            bucket = gcs_client.bucket("models_bertoxic")
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
            "input_ids": np.array(inputs["input_ids"], dtype=np.int64),
            "attention_mask": np.array(inputs["attention_mask"], dtype=np.int64),
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

        response = ToxicCommentResponse(
            text=request.text,
            label=label_name,
            confidence=confidence,
            is_toxic=bool(pred_idx == 1),
        )
        add_to_db(response)
        return response

    except Exception as e:
        traceback.print_stack()
        print(e)
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
