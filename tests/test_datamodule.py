# tests/test_datamodule.py
import pytest
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import tempfile

# Add src to path
import sys

sys.path.insert(0, Path(__file__).parent.parent / 'src')

from toxic_comments.datamodule import ToxicCommentsDataset, ToxicCommentsDataModule


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with dummy CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy CSV files
        dummy_data = pd.DataFrame({'comment_text': ['test1', 'test2', 'test3'], 'toxic': [0, 1, 0]})

        (tmp_path / 'train.csv').write_text(dummy_data.to_csv(index=False))
        (tmp_path / 'validation.csv').write_text(dummy_data.to_csv(index=False))
        (tmp_path / 'test.csv').write_text(dummy_data.to_csv(index=False))

        yield tmp_path


class TestToxicCommentsDataset:
    """Test suite for ToxicCommentsDataset class."""

    @pytest.fixture
    def mock_csv_data(self):
        """Create a mock CSV with test data."""
        return pd.DataFrame(
            {
                'comment_text': ['This is a toxic comment', 'This is a normal comment', 'Another toxic one here'],
                'toxic': [1, 0, 1],
            }
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3793, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]]),
        }
        tokenizer.side_effect = lambda *args, **kwargs: {
            'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3793, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]]),
        }
        return tokenizer

    def test_dataset_initialization(self, mock_csv_data, mock_tokenizer):
        """Test dataset can be initialized."""
        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset(csv_file='dummy.csv', tokenizer=mock_tokenizer, max_length=128)

            assert dataset is not None
            assert dataset.tokenizer == mock_tokenizer
            assert dataset.max_length == 128

    def test_dataset_length(self, mock_csv_data, mock_tokenizer):
        """Test dataset returns correct length."""
        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset('dummy.csv', mock_tokenizer)
            assert len(dataset) == 3

    def test_getitem_returns_correct_keys(self, mock_csv_data, mock_tokenizer):
        """Test __getitem__ returns dict with correct keys."""
        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset('dummy.csv', mock_tokenizer)
            item = dataset[0]

            assert isinstance(item, dict)
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'labels' in item

    def test_getitem_returns_tensors(self, mock_csv_data, mock_tokenizer):
        """Test __getitem__ returns torch tensors."""
        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset('dummy.csv', mock_tokenizer)
            item = dataset[0]

            assert isinstance(item['input_ids'], torch.Tensor)
            assert isinstance(item['attention_mask'], torch.Tensor)
            assert isinstance(item['labels'], torch.Tensor)

    def test_getitem_correct_dtypes(self, mock_csv_data, mock_tokenizer):
        """Test tensors have correct dtypes."""
        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset('dummy.csv', mock_tokenizer)
            item = dataset[0]

            assert item['input_ids'].dtype == torch.long
            assert item['attention_mask'].dtype == torch.long
            assert item['labels'].dtype == torch.long

    def test_getitem_calls_tokenizer(self, mock_csv_data):
        """Test tokenizer is called with correct arguments."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
        }

        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset('dummy.csv', mock_tokenizer, max_length=64)
            _ = dataset[0]

            mock_tokenizer.assert_called_once()
            call_kwargs = mock_tokenizer.call_args[1]

            assert call_kwargs['add_special_tokens'] is True
            assert call_kwargs['max_length'] == 64
            assert call_kwargs['padding'] == 'max_length'
            assert call_kwargs['truncation'] is True
            assert call_kwargs['return_attention_mask'] is True
            assert call_kwargs['return_tensors'] == 'pt'

    def test_getitem_with_different_indices(self, mock_csv_data, mock_tokenizer):
        """Test __getitem__ works with different indices."""
        with patch('pandas.read_csv', return_value=mock_csv_data):
            dataset = ToxicCommentsDataset('dummy.csv', mock_tokenizer)

            # Test first item
            item0 = dataset[0]
            assert item0['labels'].item() == 1

            # Test second item
            item1 = dataset[1]
            assert item1['labels'].item() == 0


class TestToxicCommentsDataModule:
    """Test suite for ToxicCommentsDataModule class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.from_pretrained.return_value = Mock()
        return tokenizer

    def test_datamodule_initialization_default(self):
        """Test DataModule initialization with default parameters."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            datamodule = ToxicCommentsDataModule()

            assert datamodule.data_dir == Path('data')
            assert datamodule.model_name_or_path == 'vinai/bertweet-base'
            assert datamodule.train_batch_size == 32
            assert datamodule.eval_batch_size == 32
            assert datamodule.max_length == 128
            assert datamodule.num_workers == 0
            mock_tokenizer.assert_called_once_with('vinai/bertweet-base', use_fast=True, normalization=True)

    def test_datamodule_initialization_custom(self):
        """Test DataModule initialization with custom parameters."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            datamodule = ToxicCommentsDataModule(
                data_dir='custom_data',
                model_name_or_path='bert-base-uncased',
                train_batch_size=16,
                eval_batch_size=8,
                max_length=256,
                num_workers=4,
            )

            assert datamodule.data_dir == Path('custom_data')
            assert datamodule.model_name_or_path == 'bert-base-uncased'
            assert datamodule.train_batch_size == 16
            assert datamodule.eval_batch_size == 8
            assert datamodule.max_length == 256
            assert datamodule.num_workers == 4
            mock_tokenizer.assert_called_once_with('bert-base-uncased', use_fast=True, normalization=True)

    def test_setup_fit_stage(self, temp_data_dir):
        """Test setup method for 'fit' stage."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer_class.return_value = mock_tokenizer

            datamodule = ToxicCommentsDataModule(data_dir=str(temp_data_dir))
            datamodule.setup('fit')

            assert datamodule.train_dataset is not None
            assert datamodule.val_dataset is not None
            assert datamodule.test_dataset is None
            assert len(datamodule.train_dataset) == 3
            assert len(datamodule.val_dataset) == 3

    def test_setup_test_stage(self, temp_data_dir):
        """Test setup method for 'test' stage."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer_class.return_value = mock_tokenizer

            datamodule = ToxicCommentsDataModule(data_dir=str(temp_data_dir))
            datamodule.setup('test')

            assert datamodule.train_dataset is None
            assert datamodule.val_dataset is None
            assert datamodule.test_dataset is not None
            assert len(datamodule.test_dataset) == 3

    def test_setup_none_stage(self, temp_data_dir):
        """Test setup method with stage=None."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer_class.return_value = mock_tokenizer

            datamodule = ToxicCommentsDataModule(data_dir=str(temp_data_dir))
            datamodule.setup()

            assert datamodule.train_dataset is not None
            assert datamodule.val_dataset is not None
            assert datamodule.test_dataset is not None

    def test_train_dataloader(self, temp_data_dir):
        """Test train_dataloader returns DataLoader with correct config."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer_class.return_value = mock_tokenizer

            datamodule = ToxicCommentsDataModule(data_dir=str(temp_data_dir), train_batch_size=4, num_workers=2)
            datamodule.setup('fit')

            train_loader = datamodule.train_dataloader()

            assert train_loader.batch_size == 4
            assert train_loader.num_workers == 2

    def test_val_dataloader(self, temp_data_dir):
        """Test val_dataloader returns DataLoader with correct config."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer_class.return_value = mock_tokenizer

            datamodule = ToxicCommentsDataModule(data_dir=str(temp_data_dir), eval_batch_size=8, num_workers=1)
            datamodule.setup('fit')

            val_loader = datamodule.val_dataloader()

            assert val_loader.batch_size == 8
            assert val_loader.num_workers == 1

    def test_test_dataloader(self, temp_data_dir):
        """Test test_dataloader returns DataLoader with correct config."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer_class.return_value = mock_tokenizer

            datamodule = ToxicCommentsDataModule(data_dir=str(temp_data_dir), eval_batch_size=16, num_workers=0)
            datamodule.setup('test')

            test_loader = datamodule.test_dataloader()

            assert test_loader.batch_size == 16
            assert test_loader.num_workers == 0


class TestIntegration:
    """Integration tests for the full data pipeline."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        return {
            'input_ids': torch.randint(0, 1000, (4, 128)),
            'attention_mask': torch.ones((4, 128)),
            'labels': torch.tensor([0, 1, 0, 1]),
        }

    def test_full_pipeline_with_mocks(self, temp_data_dir):
        """Test the full pipeline: dataset -> datamodule -> dataloader -> batch."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            # Create a realistic mock tokenizer
            mock_tokenizer = Mock()

            def mock_tokenize(text, **kwargs):
                # Simple mock: create dummy tensors based on text length
                tokens = (
                    [101] + [i % 1000 for i in range(min(len(text.split()), kwargs.get('max_length', 128) - 2))] + [102]
                )
                padded = tokens + [0] * (kwargs.get('max_length', 128) - len(tokens))

                return {
                    'input_ids': torch.tensor([padded]),
                    'attention_mask': torch.tensor(
                        [[1] * len(tokens) + [0] * (kwargs.get('max_length', 128) - len(tokens))]
                    ),
                }

            mock_tokenizer.side_effect = mock_tokenize
            mock_tokenizer_class.return_value = mock_tokenizer

            # Create and setup datamodule
            datamodule = ToxicCommentsDataModule(
                data_dir=str(temp_data_dir), train_batch_size=2, eval_batch_size=2, max_length=64
            )

            datamodule.setup('fit')

            # Get dataloader and batch
            train_loader = datamodule.train_dataloader()
            batch = next(iter(train_loader))

            # Verify batch structure
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch

            # Verify shapes
            assert batch['input_ids'].shape == (2, 64)  # batch_size x max_length
            assert batch['attention_mask'].shape == (2, 64)
            assert batch['labels'].shape == (2,)

            # Verify dtypes
            assert batch['input_ids'].dtype == torch.long
            assert batch['attention_mask'].dtype == torch.long
            assert batch['labels'].dtype == torch.long

            # Verify labels are 0 or 1
            assert torch.all((batch['labels'] == 0) | (batch['labels'] == 1))

    def test_different_batch_sizes(self, temp_data_dir):
        """Test that different batch sizes work correctly."""
        batch_sizes = [1, 2, 4, 8]

        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.side_effect = lambda *args, **kwargs: {
                'input_ids': torch.tensor([[101, 2023, 102] + [0] * 125]),
                'attention_mask': torch.tensor([[1, 1, 1] + [0] * 125]),
            }
            mock_tokenizer_class.return_value = mock_tokenizer

            for batch_size in batch_sizes:
                datamodule = ToxicCommentsDataModule(
                    data_dir=str(temp_data_dir), train_batch_size=batch_size, eval_batch_size=batch_size
                )
                datamodule.setup('fit')

                train_loader = datamodule.train_dataloader()
                batch = next(iter(train_loader))

                assert batch['input_ids'].shape[0] == min(batch_size, 3)  # We have 3 samples
                assert batch['labels'].shape[0] == min(batch_size, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
