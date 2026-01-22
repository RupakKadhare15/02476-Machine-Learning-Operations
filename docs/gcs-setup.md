# Google Cloud Storage Setup

This document describes how to set up and use Google Cloud Storage (GCS) with the toxic comments API for data drift monitoring.

## Prerequisites

1. Google Cloud project with billing enabled
2. Google Cloud SDK installed and authenticated (`gcloud auth application-default login`)
3. Required GCS buckets created:
   - `tweets-dataset` - for training data and GLOVE embeddings
   - `predictions_db` - for storing prediction logs

## Creating GCS Buckets

```bash
# Create data bucket
gsutil mb -l us-central1 gs://tweets-dataset

# Create predictions bucket
gsutil mb -l us-central1 gs://predictions_db
```

## Uploading Data to GCS

### Step 1: Compute GLOVE Embeddings Locally

First, generate the training embeddings from your local data:

```bash
uv run python src/toxic_comments/data_drift.py --compute-embeddings --data-dir data
```

This will:
- Download GLOVE 6B.50d word vectors (~163MB)
- Load the training data from `data/train.csv`
- Compute sentence embeddings (120,000 samples × 50 dimensions)
- Save embeddings to `data/train_embeddings.csv` (~109MB)

### Step 2: Upload Files to GCS

Use the provided upload script to transfer files to GCS:

```bash
# Upload all files including GLOVE vectors
uv run python scripts/upload_to_gcs.py --data-dir data

# Skip GLOVE upload if already uploaded (saves time)
uv run python scripts/upload_to_gcs.py --data-dir data --skip-glove
```

This uploads:
- `train.csv`, `validation.csv`, `test.csv` → `gs://tweets-dataset/data/`
- `train_embeddings.csv` → `gs://tweets-dataset/data/`
- `glove.6B.50d.txt` → `gs://tweets-dataset/data/`
- Creates empty `predictions.csv` → `gs://predictions_db/`

## API Configuration

The API automatically connects to GCS buckets when deployed. No configuration needed if bucket names are:
- `tweets-dataset` for data
- `predictions_db` for predictions

To use different bucket names, modify these constants in [`src/toxic_comments/api.py`](../src/toxic_comments/api.py):

```python
DATA_BUCKET_NAME = 'your-data-bucket-name'
PREDICTIONS_BUCKET_NAME = 'your-predictions-bucket-name'
```

## API Endpoints

### `/predict` - Make Predictions

Classifies a comment and saves the prediction to GCS:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test comment"}'
```

Response:
```json
{
  "text": "This is a test comment",
  "label": "non-toxic",
  "confidence": 0.95,
  "is_toxic": false
}
```

### `/monitoring` - Data Drift Detection

Generates an HTML report comparing recent predictions to training data:

```bash
# Monitor all predictions
curl "http://localhost:8000/monitoring" > monitoring_report.html

# Monitor last 100 predictions
curl "http://localhost:8000/monitoring?n=100" > monitoring_report.html
```

The report includes:
- Data drift detection on 50 embedding dimensions
- Cosine similarity score between reference and current data
- Statistical tests for each dimension
- Visual charts and distributions

## How It Works

### Prediction Storage

1. User sends a comment to `/predict`
2. API makes a prediction using the BERT model
3. Prediction is appended to `gs://predictions_db/predictions.csv`
4. Format: `time, comment_text, toxic`

### Drift Detection

1. User requests `/monitoring` endpoint
2. API downloads files from GCS to temporary directory:
   - Training embeddings from `tweets-dataset`
   - Predictions history from `predictions_db`
   - GLOVE vectors from `tweets-dataset`
3. Computes embeddings for recent predictions using GLOVE
4. Runs Evidently's DataDriftPreset on embedding columns
5. Calculates cosine similarity between reference and current embeddings
6. Returns HTML report with drift analysis
7. Cleans up temporary files

## Monitoring Strategy

The system uses **embedding-based drift detection** because:
- Text data can't be directly compared with traditional drift detection
- GLOVE embeddings convert text to 50-dimensional numerical vectors
- Drift detection operates on these numerical features
- Cosine similarity provides an overall semantic similarity metric

## Cost Considerations

GCS operations incur costs:
- **Storage**: ~$0.02/GB/month for standard storage
- **Operations**: ~$0.005 per 10,000 Class B operations (downloads)
- **Data transfer**: Free within same region

Estimated costs for typical usage:
- Storage: ~$0.005/month (270MB total)
- Predictions: ~$0.0001 per prediction (append operation)
- Monitoring: ~$0.002 per report (3 downloads)

## Troubleshooting

### Authentication Issues

```bash
# Set up application default credentials
gcloud auth application-default login

# Or use a service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Missing Files Error

If you see "Training embeddings not found", run:

```bash
uv run python src/toxic_comments/data_drift.py --compute-embeddings --data-dir data
uv run python scripts/upload_to_gcs.py --data-dir data
```

### Slow Monitoring

The monitoring endpoint downloads ~272MB of data. To optimize:
- Keep GLOVE vectors cached in the container
- Use a faster storage class (e.g., Standard vs Nearline)
- Reduce the number of reference embeddings used

## Local Development

For local testing without GCS, use mock objects:

```python
from unittest.mock import Mock, patch

with patch('src.toxic_comments.api.gcs_client') as mock_client:
    # Mock bucket operations
    mock_bucket = Mock()
    mock_blob = Mock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Test API endpoints
    response = client.post('/predict', json={'text': 'test'})
```

## Production Deployment

When deploying to production:

1. **Use workload identity** for GKE/Cloud Run (no keys needed)
2. **Enable versioning** on buckets for data safety
3. **Set lifecycle policies** to archive old predictions
4. **Monitor costs** with GCP billing alerts
5. **Use regional buckets** matching your compute region

Example bucket lifecycle policy (archive predictions after 90 days):

```bash
gsutil lifecycle set lifecycle.json gs://predictions_db
```

`lifecycle.json`:
```json
{
  "rule": [{
    "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
    "condition": {"age": 90}
  }]
}
```
