# tests/test_model.py
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, Path(__file__).parent.parent / 'src')

from toxic_comments.model import ToxicCommentsTransformer


class TestForwardPass:
    """Test suite for forward pass."""

    @pytest.fixture
    def mock_model(self):
        """Create a mocked model for testing."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model_class,
        ):
            # Create mock model with forward method
            mock_model_instance = Mock()
            mock_model_class.return_value = mock_model_instance
            mock_config.return_value = Mock()

            model = ToxicCommentsTransformer(model_name_or_path='vinai/bertweet-base', num_labels=2)

            # Store the mock model instance for assertions
            model.mock_model_instance = mock_model_instance
            return model

    def test_forward_calls_model(self, mock_model):
        """Test forward method calls the underlying model."""
        # Setup mock return value
        mock_output = Mock()
        mock_model.mock_model_instance.return_value = mock_output

        # Call forward with test inputs
        test_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'labels': torch.tensor([0]),
        }

        output = mock_model(**test_inputs)

        # Check model was called with correct inputs
        mock_model.mock_model_instance.assert_called_once_with(**test_inputs)
        assert output == mock_output

    def test_forward_returns_model_output(self, mock_model):
        """Test forward returns the model's output."""
        # Create a realistic mock output
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 2)  # batch_size=1, num_labels=2
        mock_model.mock_model_instance.return_value = mock_output

        test_inputs = {'input_ids': torch.tensor([[1, 2, 3]])}
        output = mock_model(**test_inputs)

        assert output == mock_output

    def test_output_shape(self, mock_model):
        """Test model output has correct shape."""
        # Mock model returns logits with correct shape
        batch_size = 4
        num_labels = 2

        mock_output = Mock()
        mock_output.logits = torch.randn(batch_size, num_labels)
        mock_model.mock_model_instance.return_value = mock_output

        test_inputs = {
            'input_ids': torch.randint(0, 1000, (batch_size, 128)),
            'attention_mask': torch.ones((batch_size, 128)),
        }

        output = mock_model(**test_inputs)

        assert output.logits.shape == (batch_size, num_labels)

    def test_forward_with_different_input_formats(self, mock_model):
        """Test forward accepts various input formats."""
        test_cases = [
            # (inputs dict, expected call)
            ({'input_ids': torch.tensor([[1, 2, 3]])}, {'input_ids': torch.tensor([[1, 2, 3]])}),
            (
                {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]]),
                    'labels': torch.tensor([0]),
                },
                {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]]),
                    'labels': torch.tensor([0]),
                },
            ),
            (
                {'input_ids': torch.tensor([[1, 2, 3, 4, 5]]), 'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])},
                {'input_ids': torch.tensor([[1, 2, 3, 4, 5]]), 'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])},
            ),
        ]

        for inputs, expected_call in test_cases:
            mock_model.mock_model_instance.reset_mock()

            mock_output = Mock()
            mock_model.mock_model_instance.return_value = mock_output

            output = mock_model(**inputs)

            # Check model was called with correct arguments
            call_kwargs = mock_model.mock_model_instance.call_args[1]

            for key, expected_value in expected_call.items():
                assert key in call_kwargs
                assert torch.equal(call_kwargs[key], expected_value)

            assert output == mock_output


class TestModelInitialization:
    """Test suite for model initialization."""

    def test_basic_initialization(self):
        """Test model can be instantiated with default parameters."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model,
        ):
            # Mock return values
            mock_config.return_value = Mock()
            mock_model.return_value = Mock()

            model = ToxicCommentsTransformer(model_name_or_path='vinai/bertweet-base', num_labels=2)

            assert model is not None
            assert model.hparams.model_name_or_path == 'vinai/bertweet-base'
            assert model.hparams.num_labels == 2
            assert model.hparams.learning_rate == 2e-5
            assert model.hparams.adam_epsilon == 1e-8

    def test_custom_hyperparameters(self):
        """Test model can be instantiated with custom hyperparameters."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model,
        ):
            mock_config.return_value = Mock()
            mock_model.return_value = Mock()

            model = ToxicCommentsTransformer(
                model_name_or_path='bert-base-uncased', num_labels=3, learning_rate=1e-4, adam_epsilon=1e-6
            )

            assert model.hparams.model_name_or_path == 'bert-base-uncased'
            assert model.hparams.num_labels == 3
            assert model.hparams.learning_rate == 1e-4
            assert model.hparams.adam_epsilon == 1e-6

    def test_config_loaded_correctly(self):
        """Test config is loaded with correct id2label and label2id."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model,
        ):
            # Track what config was called with
            config_kwargs = {}

            def config_side_effect(*args, **kwargs):
                config_kwargs.update(kwargs)
                config = Mock()
                config.id2label = kwargs.get('id2label', {})
                config.label2id = kwargs.get('label2id', {})
                return config

            mock_config.side_effect = config_side_effect
            mock_model.return_value = Mock()

            model = ToxicCommentsTransformer(model_name_or_path='vinai/bertweet-base', num_labels=2)

            # Check config was called with correct parameters
            assert config_kwargs['num_labels'] == 2
            assert config_kwargs['id2label'] == {0: 'NON-TOXIC', 1: 'TOXIC'}
            assert config_kwargs['label2id'] == {'NON-TOXIC': 0, 'TOXIC': 1}

            # Check model config has the mappings
            assert model.config.id2label == {0: 'NON-TOXIC', 1: 'TOXIC'}
            assert model.config.label2id == {'NON-TOXIC': 0, 'TOXIC': 1}

    def test_model_loaded_with_config(self):
        """Test pretrained model is loaded with correct config."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model,
        ):
            mock_config_instance = Mock()
            mock_config.return_value = mock_config_instance

            model = ToxicCommentsTransformer(model_name_or_path='vinai/bertweet-base', num_labels=2)

            # Check model was called with correct config
            mock_model.assert_called_once_with('vinai/bertweet-base', config=mock_config_instance)
            assert model.model == mock_model.return_value

    def test_model_in_training_mode(self):
        """Test model is set to training mode after initialization."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model,
        ):
            mock_model_instance = Mock()
            mock_model_instance.train = Mock()
            mock_model.return_value = mock_model_instance
            mock_config.return_value = Mock()

            model = ToxicCommentsTransformer(model_name_or_path='vinai/bertweet-base', num_labels=2)

            # Check model.train() was called
            mock_model_instance.train.assert_called_once()

    def test_different_model_names(self):
        """Test model works with different transformer models."""
        model_names = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']

        for model_name in model_names:
            with (
                patch('transformers.AutoConfig.from_pretrained') as mock_config,
                patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model,
            ):
                mock_config.return_value = Mock()
                mock_model.return_value = Mock()

                model = ToxicCommentsTransformer(model_name_or_path=model_name, num_labels=2)

                assert model.hparams.model_name_or_path == model_name
                mock_config.assert_called_with(
                    model_name,
                    num_labels=2,
                    id2label={0: 'NON-TOXIC', 1: 'TOXIC'},
                    label2id={'NON-TOXIC': 0, 'TOXIC': 1},
                )


class TestTrainingStep:
    """Test suite for training step."""

    @pytest.fixture
    def model_with_mock_forward(self):
        """Create model with mocked forward method."""
        with (
            patch('transformers.AutoConfig.from_pretrained') as mock_config,
            patch('transformers.AutoModelForSequenceClassification.from_pretrained'),
        ):
            mock_config.return_value = Mock()

            model = ToxicCommentsTransformer(model_name_or_path='vinai/bertweet-base', num_labels=2)

            # Mock the forward method
            model.forward = Mock()
            return model

    def test_training_step_returns_loss(self, model_with_mock_forward):
        """Test training_step returns loss from forward pass."""
        model = model_with_mock_forward

        # Setup mock return value
        mock_loss = torch.tensor(0.5, requires_grad=True)
        model.forward.return_value = (mock_loss, Mock())  # (loss, logits, ...)

        # Create test batch
        batch = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]]),
            'labels': torch.tensor([0]),
        }

        loss = model.training_step(batch, batch_idx=0)

        # Check forward was called with batch
        model.forward.assert_called_once_with(**batch)

        # Check returned loss
        assert torch.equal(loss, mock_loss)

    def test_training_step_loss_requires_grad(self, model_with_mock_forward):
        """Test loss requires gradient for backpropagation."""
        model = model_with_mock_forward

        # Create loss that requires grad
        mock_loss = torch.tensor(0.3, requires_grad=True)
        model.forward.return_value = (mock_loss, Mock())

        batch = {'input_ids': torch.tensor([[1, 2, 3]])}
        loss = model.training_step(batch, batch_idx=0)

        assert loss.requires_grad

    def test_training_step_with_different_batch_sizes(self, model_with_mock_forward):
        """Test training_step handles different batch sizes."""
        model = model_with_mock_forward

        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            model.forward.reset_mock()

            # Create mock loss
            mock_loss = torch.tensor(0.1 * batch_size, requires_grad=True)
            model.forward.return_value = (mock_loss, Mock())

            # Create batch with specific size
            batch = {
                'input_ids': torch.randint(0, 1000, (batch_size, 128)),
                'attention_mask': torch.ones((batch_size, 128)),
                'labels': torch.randint(0, 2, (batch_size,)),
            }

            loss = model.training_step(batch, batch_idx=0)

            # Check forward was called
            model.forward.assert_called_once()

            # Check batch was passed correctly
            call_kwargs = model.forward.call_args[1]
            assert call_kwargs['input_ids'].shape[0] == batch_size
            assert call_kwargs['attention_mask'].shape[0] == batch_size
            assert call_kwargs['labels'].shape[0] == batch_size

            assert torch.equal(loss, mock_loss)

    def test_training_step_batch_idx_ignored(self, model_with_mock_forward):
        """Test training_step ignores batch_idx parameter (just receives it)."""
        model = model_with_mock_forward

        mock_loss = torch.tensor(0.2, requires_grad=True)
        model.forward.return_value = (mock_loss, Mock())

        batch = {'input_ids': torch.tensor([[1, 2, 3]])}

        # Test with different batch_idx values
        for batch_idx in [0, 1, 10, 100]:
            model.forward.reset_mock()
            loss = model.training_step(batch, batch_idx=batch_idx)

            # Should still call forward with same batch
            model.forward.assert_called_once_with(**batch)
            assert torch.equal(loss, mock_loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
