"""
Data drift monitoring script for toxic comments classifier.

This script:
1. Downloads the Ruddit dataset from Kaggle
2. Runs predictions on a sample of comments via the API
3. Generates drift monitoring reports
4. Calculates performance metrics (accuracy, F1, AUC)
"""

from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# API endpoint configuration
API_BASE_URL = 'http://localhost:8000'
DATA_DIR = 'data'
PRED_DB_PATH = '/tmp/drift_predictions_db'


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

    args = parser.parse_args()

    if args.download_only:
        download_ruddit_dataset()
    else:
        print('Starting drift monitoring pipeline...')
        print(f'API URL: {args.api_url}')
        print(f'Sample size: {args.sample_size}')
        run_predictions_and_monitoring(sample_size=args.sample_size, api_url=args.api_url, data_dir=args.data_dir)
