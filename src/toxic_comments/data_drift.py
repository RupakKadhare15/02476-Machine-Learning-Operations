"""
Data drift monitoring script for toxic comments classifier.

This script:
1. Downloads the Ruddit dataset from Kaggle
2. Downloads and processes GLOVE embeddings for text data
3. Runs predictions on a sample of comments via the API
4. Generates drift monitoring reports using embedding drift detection
5. Calculates performance metrics (accuracy, F1, AUC)
"""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# API endpoint configuration
API_BASE_URL = 'http://localhost:8000'
DATA_DIR = 'data'
PRED_DB_PATH = '/tmp/drift_predictions_db'
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_DIM = 50


def download_glove_vectors(data_dir: str = DATA_DIR):
    """
    Download and extract GLOVE vectors if not already present.

    Args:
        data_dir: Directory to save GLOVE vectors

    """
    glove_file = Path(data_dir) / f'glove.6B.{GLOVE_DIM}d.txt'

    if glove_file.exists():
        print(f'GLOVE vectors already exist at {glove_file}')
        return glove_file

    print('Downloading GLOVE vectors...')
    zip_path = Path(data_dir) / 'glove.6B.zip'
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the zip file with progress bar
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc='Downloading') as t:
        urllib.request.urlretrieve(GLOVE_URL, zip_path, reporthook=t.update_to)

    print('Download complete. Extracting...')

    # Extract only the file we need
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc='Extracting'):
            if member == f'glove.6B.{GLOVE_DIM}d.txt':
                zip_ref.extract(member, data_dir)
                break

    # Remove the zip file to save space
    zip_path.unlink()
    print(f'GLOVE vectors saved to {glove_file}')

    return glove_file


def load_glove_model(glove_file: Path):
    """
    Load GLOVE vectors from a text file.

    Args:
        glove_file: Path to the GLOVE vectors file

    Returns:
        Dictionary mapping words to their vector representations

    """
    print(f'Loading GLOVE model from {glove_file}...')
    glove_model = {}

    # Count lines for progress bar
    with open(glove_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_lines, desc='Loading GLOVE vectors'):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding

    print(f'Loaded {len(glove_model)} word vectors')
    return glove_model


def get_sentence_embeddings(texts, glove_model, dim=GLOVE_DIM):
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

    for text in tqdm(texts, desc='Computing embeddings'):
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


def compute_and_save_train_embeddings(data_dir: str = DATA_DIR):
    """
    Compute GLOVE embeddings for training data and save them.

    Args:
        data_dir: Directory containing training data and to save embeddings

    """
    train_path = Path(data_dir) / 'train.csv'
    embeddings_path = Path(data_dir) / 'train_embeddings.csv'

    if embeddings_path.exists():
        print(f'Training embeddings already exist at {embeddings_path}')
        return embeddings_path

    if not train_path.exists():
        print(f'Training data not found at {train_path}')
        return None

    print('Computing embeddings for training data...')

    # Download GLOVE vectors if needed
    glove_file = download_glove_vectors(data_dir)
    glove_model = load_glove_model(glove_file)

    # Load training data
    train_df = pd.read_csv(train_path)
    print(f'Loaded {len(train_df)} training samples')

    # Compute embeddings
    embeddings = get_sentence_embeddings(train_df['comment_text'], glove_model)

    # Save as DataFrame
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [f'emb_{i}' for i in range(embeddings_df.shape[1])]
    embeddings_df.to_csv(embeddings_path, index=False)

    print(f'Training embeddings saved to {embeddings_path}')
    return embeddings_path


def download_ruddit_dataset():
    """Download and prepare the Ruddit dataset from Kaggle."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        print('Downloading Ruddit dataset from Kaggle...')
        file_path = 'ruddit_comments_score.csv'

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            'estebanmarcelloni/ruddit-papers-comments-scored',
            file_path,
        )

        # Convert score to binary toxic label (score > 1 means toxic)
        df['toxic'] = (df['score'] > 1).astype(int)

        # Save to data directory
        output_path = Path(DATA_DIR) / 'ruddit_comments_score.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f'Dataset saved to {output_path}')
        return df
    except ImportError:
        print('Error: kagglehub not installed. Install with: pip install kagglehub[pandas-datasets]')
        return None
    except Exception as e:
        print(f'Error downloading dataset: {e}')
        return None


def run_predictions_and_monitoring(sample_size: int = 5000, api_url: str = API_BASE_URL, data_dir: str = DATA_DIR):
    """
    Run predictions on Ruddit dataset and generate drift monitoring report.

    Args:
        sample_size: Number of comments to use for drift monitoring
        api_url: Base URL of the API
        data_dir: Directory containing the dataset

    """
    # Load the Ruddit comments dataset
    ruddit_path = Path(data_dir) / 'ruddit_comments_score.csv'
    if not ruddit_path.exists():
        print(f'Dataset not found at {ruddit_path}. Downloading...')
        df = download_ruddit_dataset()
        if df is None:
            print('Failed to download dataset. Exiting.')
            return
    else:
        df = pd.read_csv(ruddit_path)

    print(f'Loaded {len(df)} comments from Ruddit dataset')

    # Take a sample
    sample_comments = df.head(min(sample_size, len(df)))
    print(f'Using {len(sample_comments)} comments for monitoring')

    # Collect predictions and metrics
    y_true = []
    y_pred = []
    y_pred_proba = []
    successful_predictions = 0
    failed_predictions = 0

    print('\nMaking predictions via API...')
    for idx, row in sample_comments.iterrows():
        try:
            response = requests.post(
                f'{api_url}/predict',
                json={'text': str(row['body'])},
                params={'pred_db_path': PRED_DB_PATH},
                timeout=30,
            )
            if response.status_code == 200:
                pred_data = response.json()
                # Store true label
                y_true.append(int(row['toxic']))
                # Store predicted label
                y_pred.append(1 if pred_data['is_toxic'] else 0)
                # Store prediction confidence
                y_pred_proba.append(pred_data['confidence'])
                successful_predictions += 1

                if successful_predictions % 100 == 0:
                    print(f'  Processed {successful_predictions} predictions...')
            else:
                failed_predictions += 1
        except Exception as e:
            failed_predictions += 1
            if failed_predictions <= 5:  # Only print first 5 errors
                print(f'  Failed to predict: {e}')

    print(f'\nCompleted {successful_predictions} successful predictions ({failed_predictions} failures)')

    if successful_predictions == 0:
        print('No successful predictions. Check if API is running.')
        return

    # Calculate and display metrics
    print('\n' + '=' * 60)
    print('MODEL PERFORMANCE METRICS ON RUDDIT DATASET')
    print('=' * 60)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')

    print(f'Samples evaluated: {successful_predictions}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Calculate AUC if we have both classes
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f'AUC: {auc:.4f}')
    else:
        print('AUC: N/A (insufficient class diversity)')

    print(f'\nTrue label distribution: {pd.Series(y_true).value_counts().to_dict()}')
    print(f'Predicted label distribution: {pd.Series(y_pred).value_counts().to_dict()}')
    print('=' * 60)

    # Generate drift monitoring report
    print('\nGenerating drift monitoring report...')
    try:
        response = requests.get(
            f'{api_url}/monitoring',
            params={'data_dir': data_dir, 'pred_db_path': PRED_DB_PATH, 'n': successful_predictions},
            timeout=60,
        )

        if response.status_code == 200:
            # Save the HTML report
            report_path = Path('drift_monitoring_report.html')
            report_path.write_text(response.text)
            print(f'Drift monitoring report saved to: {report_path.absolute()}')
            print('Open this file in a browser to view the detailed drift analysis.')
        else:
            print(f'Failed to generate monitoring report. Status code: {response.status_code}')
    except Exception as e:
        print(f'Error generating monitoring report: {e}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run drift monitoring on Ruddit dataset')
    parser.add_argument('--sample-size', type=int, default=5000, help='Number of comments to process (default: 5000)')
    parser.add_argument('--api-url', type=str, default=API_BASE_URL, help=f'API base URL (default: {API_BASE_URL})')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help=f'Data directory (default: {DATA_DIR})')
    parser.add_argument('--download-only', action='store_true', help='Only download the dataset and exit')
    parser.add_argument(
        '--compute-embeddings', action='store_true', help='Compute and save GLOVE embeddings for training data'
    )

    args = parser.parse_args()

    if args.compute_embeddings:
        print('Computing GLOVE embeddings for training data...')
        compute_and_save_train_embeddings(args.data_dir)
    elif args.download_only:
        download_ruddit_dataset()
    else:
        print('Starting drift monitoring pipeline...')
        print(f'API URL: {args.api_url}')
        print(f'Sample size: {args.sample_size}')
        run_predictions_and_monitoring(sample_size=args.sample_size, api_url=args.api_url, data_dir=args.data_dir)
