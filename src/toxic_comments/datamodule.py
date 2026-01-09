import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional


class ToxicCommentsDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length: int = 128):
        """
        Dataset for toxic comments classification.
        
        Args:
            csv_file: Path to the CSV file containing comment_text and toxic columns
            tokenizer: Hugging Face tokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ToxicCommentsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        model_name_or_path: str = "vinai/bertweet-base",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 0,
    ):
        """
        PyTorch Lightning DataModule for Toxic Comments dataset.
        
        Args:
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
            normalization=True
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ToxicCommentsDataset(
                csv_file=self.data_dir / "train.csv",
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            self.val_dataset = ToxicCommentsDataset(
                csv_file=self.data_dir / "validation.csv",
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ToxicCommentsDataset(
                csv_file=self.data_dir / "test.csv",
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    # Test the datamodule
    datamodule = ToxicCommentsDataModule(
        data_dir="data",
        model_name_or_path="vinai/bertweet-base",
        train_batch_size=8,
        eval_batch_size=8,
        max_length=128,
        num_workers=0  # Use 0 for testing to avoid multiprocessing pickle issues
    )
    
    # Setup for training
    datamodule.setup('fit')
    
    # Get a batch from training data
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Sample label: {batch['labels'][0]}")
    print(f"\nTrain dataset size: {len(datamodule.train_dataset)}")
    print(f"Validation dataset size: {len(datamodule.val_dataset)}")
