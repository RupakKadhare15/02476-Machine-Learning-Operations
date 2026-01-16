# tests/test_data.py
# Add src to path
import sys
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, Path(__file__).parent.parent / 'src')


@pytest.fixture
def mock_path():
    """Fixture providing a mock Path object."""
    mock = Mock()
    mock.mkdir = Mock()
    mock.__truediv__ = Mock(return_value=mock)
    mock.__str__ = Mock(return_value='data')
    return mock


class TestDataModule:

    """Tests for data.py module structure."""

    def test_import(self):
        """Test module can be imported."""
        from toxic_comments import data

        assert data is not None

    def test_constants_exist(self):
        """Test required constants are defined."""
        from toxic_comments import data

        assert hasattr(data, 'DRIVE_ID')
        assert hasattr(data, 'DATA_PATH')

    def test_constant_values(self):
        """Test constants have correct values."""
        from toxic_comments import data

        assert data.DRIVE_ID == '1czsN8ebcoAkwAhs6rKdw3Enz0oBzdzTP'
        assert str(data.DATA_PATH) == 'data'

    def test_main_function_exists(self):
        """Test main function exists and is callable."""
        from toxic_comments import data

        assert hasattr(data, 'main')
        assert callable(data.main)


class TestDataFunctionality:

    """Tests for data.py functionality with mocks."""

    def test_directory_creation(self, mock_path):
        """Test that data directory is created."""
        from toxic_comments import data

        with patch('toxic_comments.data.DATA_PATH', mock_path):
            with patch('toxic_comments.data.requests.Session'):
                with patch('toxic_comments.data.shutil.unpack_archive'):
                    with patch('builtins.open', mock_open()):
                        data.main()

        mock_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_google_drive_api_call(self, mock_path):
        """Test correct Google Drive API call."""
        from toxic_comments import data

        with patch('toxic_comments.data.DATA_PATH', mock_path):
            with patch('toxic_comments.data.requests.Session') as mock_session_class:
                with patch('toxic_comments.data.shutil.unpack_archive'):
                    with patch('builtins.open', mock_open()):
                        mock_session = Mock()
                        mock_response = Mock()
                        mock_response.cookies = {}
                        mock_response.iter_content.return_value = []
                        mock_session.get.return_value = mock_response
                        mock_session_class.return_value = mock_session

                        data.main()

                        mock_session.get.assert_called_once_with(
                            'https://docs.google.com/uc?export=download', params={'id': data.DRIVE_ID}, stream=True
                        )

    def test_archive_extraction(self, mock_path):
        """Test zip file extraction."""
        from toxic_comments import data

        with patch('toxic_comments.data.DATA_PATH', mock_path):
            with patch('toxic_comments.data.requests.Session'):
                with patch('toxic_comments.data.shutil.unpack_archive') as mock_unpack:
                    with patch('builtins.open', mock_open()):
                        data.main()

                        mock_unpack.assert_called_once_with(
                            mock_path,  # data.zip path
                            mock_path,  # DATA_PATH
                        )


# Data directory for file tests
DATA_DIR = Path(__file__).parent.parent / 'data'


@pytest.mark.skipif(
    not all((DATA_DIR / f).exists() for f in ['test.csv', 'train.csv', 'validation.csv']),
    reason='Run data.py to download data first',
)
class TestDataFiles:

    """Tests for downloaded data files."""

    @pytest.fixture(params=['test.csv', 'train.csv', 'validation.csv'])
    def data_df(self, request):
        """Fixture providing each data file as DataFrame."""
        return pd.read_csv(DATA_DIR / request.param), request.param

    def test_has_two_columns(self, data_df):
        """Test each file has exactly 2 columns."""
        df, filename = data_df
        assert len(df.columns) == 2

    def test_column_types(self, data_df):
        """Test column data types."""
        df, _ = data_df
        assert df.iloc[:, 0].dtype == 'object'  # Text column is string

        # Label column should be integer type (int64, int32, etc.)
        assert np.issubdtype(df.iloc[:, 1].dtype, np.integer)

    def test_binary_labels(self, data_df):
        """Test labels are only 0 or 1."""
        df, _ = data_df
        labels = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        unique = set(labels.dropna().unique())
        assert unique.issubset({0, 1})

    def test_no_nan_values(self, data_df):
        """Test no NaN values."""
        df, _ = data_df
        for col in df.columns:
            assert df[col].isna().sum() == 0

    def test_text_not_empty(self, data_df):
        """Test text column has no empty strings."""
        df, _ = data_df
        text_col = df.iloc[:, 0]
        empty_count = text_col.apply(lambda x: str(x).strip() == '').sum()
        assert empty_count == 0

    def test_train_data_has_both_classes(self):
        """Test train.csv has both classes 0 and 1."""
        if (DATA_DIR / 'train.csv').exists():
            df = pd.read_csv(DATA_DIR / 'train.csv')
            labels = pd.to_numeric(df.iloc[:, 1], errors='coerce')
            unique = set(labels.dropna().unique())
            assert 0 in unique and 1 in unique
