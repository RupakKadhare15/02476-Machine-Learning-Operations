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
    """Test suite for forward pass (Category B)."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mocked model for testing."""
        with patch('transformers.AutoConfig.from_pretrained') as mock_config, \
             patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model_class:
            
            # Create mock model with forward method
            mock_model_instance = Mock()
            mock_model_class.return_value = mock_model_instance
            mock_config.return_value = Mock()
            
            model = ToxicCommentsTransformer(
                model_name_or_path='vinai/bertweet-base',
                num_labels=2
            )
            
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
            'labels': torch.tensor([0])
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
            'attention_mask': torch.ones((batch_size, 128))
        }
        
        output = mock_model(**test_inputs)
        
        assert output.logits.shape == (batch_size, num_labels)
    
    def test_forward_with_different_input_formats(self, mock_model):
        """Test forward accepts various input formats."""
        test_cases = [
            # (inputs dict, expected call)
            (
                {'input_ids': torch.tensor([[1, 2, 3]])},
                {'input_ids': torch.tensor([[1, 2, 3]])}
            ),
            (
                {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]]),
                    'labels': torch.tensor([0])
                },
                {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]]),
                    'labels': torch.tensor([0])
                }
            ),
            (
                {
                    'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
                },
                {
                    'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
                }
            )
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])