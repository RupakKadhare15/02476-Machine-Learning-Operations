from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ToxicCommentsDataset(Dataset):
    """Dataset for toxic comments classification."""

    def __init__(self, csv_file: str, tokenizer, max_length: int = 128):
        """
        Dataset for toxic comments classification.

        Args:
        ----
            csv_file: Path to the CSV file containing comment_text and toxic columns
            tokenizer: Hugging Face tokenizer instance
            max_length: Maximum sequence length for tokenization

        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        :param self: The dataset instance
        :return: Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        :param self: The dataset instance
        :param idx: Index of the sample to retrieve
        :return: A dictionary containing input_ids, attention_mask, and labels
        """
        text = str(self.data.iloc[idx]['comment_text'])
        label = int(self.data.iloc[idx]['toxic'])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }


class ToxicCommentsDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Toxic Comments dataset."""

    def __init__(
        self,
        data_dir: str = 'data',
        model_name_or_path: str = 'vinai/bertweet-base',
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 0,
    ):
        """
        PyTorch Lightning DataModule for Toxic Comments dataset.

        Args:
        ----
            data_dir: Directory containing train.csv, validation.csv, and test.csv
            model_name_or_path: Model identifier for tokenizer
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for validation and testing
            max_length: Maximum sequence length for tokenization
            num_workers: Number of workers for data loading

        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, normalization=True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset.

        Args:
        ----
            stage: Either 'fit', 'validate', 'test', or 'predict'

        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ToxicCommentsDataset(
                csv_file=self.data_dir / 'train.csv', tokenizer=self.tokenizer, max_length=self.max_length
            )
            self.val_dataset = ToxicCommentsDataset(
                csv_file=self.data_dir / 'validation.csv', tokenizer=self.tokenizer, max_length=self.max_length
            )

        if stage == 'test' or stage is None:
            self.test_dataset = ToxicCommentsDataset(
                csv_file=self.data_dir / 'test.csv', tokenizer=self.tokenizer, max_length=self.max_length
            )

    def train_dataloader(self):
        """Return DataLoader for training dataset."""
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Return DataLoader for validation dataset."""
        return DataLoader(
            self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return DataLoader for test dataset."""
        return DataLoader(
            self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers
        )


def dataset_statistics(data_dir: str = 'data', output_dir: str = 'reports/figures') -> dict:
    """
    Generate and report basic statistics about the Toxic Comments dataset.

    Args:
    ----
        data_dir: Directory containing train.csv, validation.csv, and test.csv
        output_dir: Directory to save generated figures

    Returns:
    -------
        dict: Dictionary containing dataset statistics

    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_df = pd.read_csv(data_path / 'train.csv')
    val_df = pd.read_csv(data_path / 'validation.csv')
    test_df = pd.read_csv(data_path / 'test.csv')

    # Calculate basic statistics
    stats = {
        'train_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_toxic': train_df['toxic'].sum(),
        'train_non_toxic': (train_df['toxic'] == 0).sum(),
        'val_toxic': val_df['toxic'].sum(),
        'val_non_toxic': (val_df['toxic'] == 0).sum(),
        'test_toxic': test_df['toxic'].sum(),
        'test_non_toxic': (test_df['toxic'] == 0).sum(),
    }

    # Calculate class balance percentages
    stats['train_toxic_pct'] = (stats['train_toxic'] / stats['train_samples']) * 100
    stats['val_toxic_pct'] = (stats['val_toxic'] / stats['validation_samples']) * 100
    stats['test_toxic_pct'] = (stats['test_toxic'] / stats['test_samples']) * 100

    # Calculate text length statistics
    train_df['text_length'] = train_df['comment_text'].str.len()
    val_df['text_length'] = val_df['comment_text'].str.len()
    test_df['text_length'] = test_df['comment_text'].str.len()

    stats['train_avg_length'] = train_df['text_length'].mean()
    stats['val_avg_length'] = val_df['text_length'].mean()
    stats['test_avg_length'] = test_df['text_length'].mean()
    stats['train_max_length'] = train_df['text_length'].max()
    stats['val_max_length'] = val_df['text_length'].max()
    stats['test_max_length'] = test_df['text_length'].max()

    # Print statistics
    print('=' * 60)
    print('TOXIC COMMENTS DATASET STATISTICS')
    print('=' * 60)
    print('\nDataset Sizes:')
    print(f'  Training samples:     {stats["train_samples"]:,}')
    print(f'  Validation samples:   {stats["validation_samples"]:,}')
    print(f'  Test samples:         {stats["test_samples"]:,}')
    print(f'  Total samples:        {stats["total_samples"]:,}')

    print('\nClass Distribution:')
    print('  Training:')
    print(f'    - Toxic:            {stats["train_toxic"]:,} ({stats["train_toxic_pct"]:.2f}%)')
    print(f'    - Non-toxic:        {stats["train_non_toxic"]:,} ({100 - stats["train_toxic_pct"]:.2f}%)')
    print('  Validation:')
    print(f'    - Toxic:            {stats["val_toxic"]:,} ({stats["val_toxic_pct"]:.2f}%)')
    print(f'    - Non-toxic:        {stats["val_non_toxic"]:,} ({100 - stats["val_toxic_pct"]:.2f}%)')
    print('  Test:')
    print(f'    - Toxic:            {stats["test_toxic"]:,} ({stats["test_toxic_pct"]:.2f}%)')
    print(f'    - Non-toxic:        {stats["test_non_toxic"]:,} ({100 - stats["test_toxic_pct"]:.2f}%)')

    print('\nText Length Statistics:')
    print(f'  Training:   Avg={stats["train_avg_length"]:.1f}, Max={stats["train_max_length"]}')
    print(f'  Validation: Avg={stats["val_avg_length"]:.1f}, Max={stats["val_max_length"]}')
    print(f'  Test:       Avg={stats["test_avg_length"]:.1f}, Max={stats["test_max_length"]}')
    print('=' * 60)

    # Set style for better-looking plots
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)

    # Figure 1: Class Distribution across splits
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    splits = ['Train', 'Validation', 'Test']
    dfs = [train_df, val_df, test_df]

    for ax, split_name, df in zip(axes, splits, dfs):
        counts = df['toxic'].value_counts().sort_index()
        colors = ['#2ecc71', '#e74c3c']  # Green for non-toxic, Red for toxic
        ax.bar(['Non-Toxic', 'Toxic'], counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(f'{split_name} Set', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Class', fontsize=12)
        # Add count labels on bars
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(counts.values) * 0.01, f'{v:,}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Class Distribution Across Dataset Splits', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f'\nSaved: {output_path / "class_distribution.png"}')
    plt.close()

    # Figure 2: Text Length Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram of text lengths for each split
    for ax, split_name, df, color in zip(axes.flat[:3], splits, dfs, ['#3498db', '#9b59b6', '#e67e22']):
        ax.hist(df['text_length'], bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(f'{split_name} - Text Length Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Text Length (characters)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.axvline(
            df['text_length'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {df["text_length"].mean():.1f}',
        )
        ax.legend()

    # Combined boxplot
    ax = axes.flat[3]
    combined_df = pd.concat(
        [
            train_df[['text_length']].assign(split='Train'),
            val_df[['text_length']].assign(split='Validation'),
            test_df[['text_length']].assign(split='Test'),
        ]
    )
    sns.boxplot(data=combined_df, x='split', y='text_length', ax=ax, hue='split', palette='Set2', legend=False)
    ax.set_title('Text Length Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset Split', fontsize=10)
    ax.set_ylabel('Text Length (characters)', fontsize=10)

    plt.suptitle('Text Length Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path / "text_length_distribution.png"}')
    plt.close()

    # Figure 3: Text Length by Class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, split_name, df in zip(axes, splits, dfs):
        toxic_lengths = df[df['toxic'] == 1]['text_length']
        non_toxic_lengths = df[df['toxic'] == 0]['text_length']

        ax.hist(
            [non_toxic_lengths, toxic_lengths],
            bins=40,
            label=['Non-Toxic', 'Toxic'],
            color=['#2ecc71', '#e74c3c'],
            alpha=0.6,
            edgecolor='black',
        )
        ax.set_title(f'{split_name} Set', fontsize=12, fontweight='bold')
        ax.set_xlabel('Text Length (characters)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()

    plt.suptitle('Text Length Distribution by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'text_length_by_class.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path / "text_length_by_class.png"}')
    plt.close()

    # Figure 4: Class Balance Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    splits_data = {
        'Train': [stats['train_non_toxic'], stats['train_toxic']],
        'Validation': [stats['val_non_toxic'], stats['val_toxic']],
        'Test': [stats['test_non_toxic'], stats['test_toxic']],
    }

    x = range(len(splits_data))
    width = 0.35

    non_toxic_counts = [v[0] for v in splits_data.values()]
    toxic_counts = [v[1] for v in splits_data.values()]

    bars1 = ax.bar([i - width / 2 for i in x], non_toxic_counts, width, label='Non-Toxic', color='#2ecc71', alpha=0.7)
    bars2 = ax.bar([i + width / 2 for i in x], toxic_counts, width, label='Toxic', color='#e74c3c', alpha=0.7)

    ax.set_xlabel('Dataset Split', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Balance Across Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits_data.keys())
    ax.legend()

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{int(height):,}',
                ha='center',
                va='bottom',
                fontweight='bold',
            )

    plt.tight_layout()
    plt.savefig(output_path / 'class_balance.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path / "class_balance.png"}')
    plt.close()

    print('\n' + '=' * 60)
    print('All figures saved successfully!')
    print('=' * 60)

    return stats


if __name__ == '__main__':
    stats = dataset_statistics(data_dir='data', output_dir='reports/figures')
