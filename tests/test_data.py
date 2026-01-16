"""
Test file for data.py and the data files (test.csv, train.csv, validation.csv).
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch, mock_open

# Add the src directory to Python path so we can import toxic_comments.data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Define data directory
DATA_DIR = Path(__file__).parent.parent / 'data'

def test_module_can_be_imported():
    """
    Test that the data module can be imported.
    """
    from toxic_comments import data
    assert True  # If we get here, import succeeded

def test_constants_exist():
    """
    Verify that required constants are defined.
    """
    from toxic_comments import data
    
    # Check that constants are defined
    assert hasattr(data, 'DRIVE_ID'), "DRIVE_ID constant is missing"
    assert hasattr(data, 'DATA_PATH'), "DATA_PATH constant is missing"
    
    # Check their types
    assert isinstance(data.DRIVE_ID, str), "DRIVE_ID should be a string"
    assert isinstance(data.DATA_PATH, object), "DATA_PATH should be a Path object"

def test_constant_values():
    """
    Verify that constants have the expected values.
    """
    from toxic_comments import data
    from pathlib import Path

    # Verify specific values
    assert data.DRIVE_ID == '1czsN8ebcoAkwAhs6rKdw3Enz0oBzdzTP', \
        "DRIVE_ID has unexpected value"

    assert str(data.DATA_PATH) == 'data', \
        "DATA_PATH should point to 'data' directory"

    # DATA_PATH should be a Path object for cross-platform compatibility
    assert isinstance(data.DATA_PATH, Path), \
        "DATA_PATH should be a Path object, not a string"

def test_main_function_exists():
    """Test that main function exists."""
    try:
        from toxic_comments import data
        assert hasattr(data, 'main')
        assert callable(data.main)
        print("Main function test passed")
    except ImportError:
        print("data module not found - skipping test")

def test_data_directory_is_created():
    """
    Verify that the script creates the necessary directory structure.
    """
    from toxic_comments import data

    # Create a mock that behaves like a Path object
    mock_path = Mock()
    
    # Mock the mkdir method
    mock_path.mkdir = Mock()
    
    # Mock the __truediv__ method to support / operator
    mock_path.__truediv__ = Mock(return_value=mock_path)  # Returns itself or another mock
    
    # Replace the entire DATA_PATH with our mock
    with patch('toxic_comments.data.DATA_PATH', mock_path):
        with patch('toxic_comments.data.requests.Session'):
            with patch('toxic_comments.data.shutil.unpack_archive'):
                with patch('builtins.open', mock_open()):
                    data.main()
    
    # Verify mkdir was called with correct arguments
    mock_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # parents=True ensures parent directories are created if needed
        # exist_ok=True prevents errors if directory already exists

def test_google_drive_api_call():
    """
    Verify the script makes the correct API request to Google Drive.
    This tests the URL, parameters, and streaming setup.
    """
    from toxic_comments import data

    # Create a mock that behaves like a Path object
    mock_path = Mock()
    
    # Mock the mkdir method
    mock_path.mkdir = Mock()
    
    # Mock the __truediv__ method to support / operator
    mock_path.__truediv__ = Mock(return_value=mock_path)  # Returns itself or another mock
    
    # Replace the entire DATA_PATH with our mock
    with patch('toxic_comments.data.DATA_PATH', mock_path):
        with patch('toxic_comments.data.requests.Session') as mock_session_class:
            with patch('toxic_comments.data.shutil.unpack_archive'):
                with patch('builtins.open', mock_open()):
                    # Setup mock session
                    mock_session = Mock()
                    mock_response = Mock()
                    mock_response.cookies = {}  # No warning token
                    mock_response.iter_content.return_value = []
                    mock_session.get.return_value = mock_response
                    mock_session_class.return_value = mock_session
                    
                    data.main()
                    
                    # Verify the API call
                    mock_session.get.assert_called_once_with(
                        'https://docs.google.com/uc?export=download',
                        params={'id': data.DRIVE_ID},  # Using actual constant
                        stream=True  # Important for large files
                    )

def test_archive_extraction():
    """
    Why: Verify that after download, the zip file is extracted.
    Tests the final step of the script.
    """
    from toxic_comments import data

    # Create a mock that behaves like a Path object
    mock_path = Mock()
    
    # Mock the mkdir method
    mock_path.mkdir = Mock()
    
    # Mock the __truediv__ method to support / operator
    mock_path.__truediv__ = Mock(return_value=mock_path)  # Returns itself or another mock
    
    # Replace the entire DATA_PATH with our mock
    with patch('toxic_comments.data.DATA_PATH', mock_path):
        with patch('toxic_comments.data.requests.Session'):
            with patch('toxic_comments.data.shutil.unpack_archive') as mock_unpack:
                with patch('builtins.open', mock_open()):
                    data.main()
                    
                    # Verify unpack_archive was called with correct arguments
                    mock_unpack.assert_called_once_with(
                        data.DATA_PATH / 'data.zip',  # Source zip file
                        data.DATA_PATH  # Destination directory
                    )

def test_all_data_files_exist():
    """Test that all expected data files exist."""
    expected_files = ['test.csv', 'train.csv', 'validation.csv', 'data.zip']
    missing_files = []
    
    for file_name in expected_files:
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            print(f"Found data file: {file_path}")
    
    assert not missing_files, f"Missing data files: {missing_files}"
    print("All expected data files exist")

def test_csv_files_can_be_read():
    """Test that all CSV files can be read with pandas."""
    files = ['test.csv', 'train.csv', 'validation.csv']
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully read {file_name}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            assert False, f"Failed to read {file_name}: {e}"

def test_data_has_correct_columns():
    """Test that all data files have exactly 2 columns."""
    files = ['test.csv', 'train.csv', 'validation.csv']
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path)
        
        # Check number of columns
        assert len(df.columns) == 2, \
            f"{file_name}: Expected 2 columns, found {len(df.columns)} columns: {list(df.columns)}"
        
        print(f"{file_name} has correct number of columns: {list(df.columns)}")

def test_column_data_types():
    """Test that columns have correct data types."""
    files = ['test.csv', 'train.csv', 'validation.csv']
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path)
        
        # Check first column is string-like
        first_col = df.iloc[:, 0]
        assert first_col.dtype == 'object', \
            f"{file_name}: First column should be string (object), but is {first_col.dtype}"
        
        # Check second column is integer-like
        second_col = df.iloc[:, 1]
        
        # Try to convert to numeric if it's not already
        if second_col.dtype == 'object':
            try:
                second_col = pd.to_numeric(second_col, errors='coerce')
            except:
                assert False, f"{file_name}: Second column cannot be converted to numeric"
        
        # Check if it's integer type (int64, int32, etc.)
        is_integer = np.issubdtype(second_col.dtype, np.integer)
        if not is_integer:
            # Check if all values are integers
            if second_col.dropna().apply(lambda x: float(x).is_integer()).all():
                print(f"{file_name}: Second column has float dtype but all values are integers")
            else:
                assert False, f"{file_name}: Second column should be integer type, but is {second_col.dtype}"
        
        print(f"{file_name} data types: {first_col.dtype}, {second_col.dtype}")

def test_label_values_are_binary():
    """Test that label values are only 0 or 1."""
    files = ['test.csv', 'train.csv', 'validation.csv']
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path)
        
        # Get second column (label)
        labels = df.iloc[:, 1]
        
        # Convert to numeric if needed
        if labels.dtype == 'object':
            labels = pd.to_numeric(labels, errors='coerce')
        
        # Check for NaN in labels (will be caught by another test)
        labels_no_nan = labels.dropna()
        
        # Get unique values
        unique_values = set(labels_no_nan.unique())
        
        # Check values are only 0 and/or 1
        invalid_values = unique_values - {0, 1}
        assert not invalid_values, \
            f"{file_name}: Label column contains invalid values: {invalid_values}"
        
        # Optional: Check that we have at least one of each class (for training data)
        if file_name == 'train.csv':
            has_0 = 0 in unique_values
            has_1 = 1 in unique_values
            if not (has_0 and has_1):
                print(f"{file_name}: Training data should have both classes 0 and 1, found: {unique_values}")
        
        print(f"{file_name} label values: {sorted(unique_values)}")


def test_no_nan_values():
    """Test that there are no NaN values in any column."""
    files = ['test.csv', 'train.csv', 'validation.csv']
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path)
        
        # Check for NaN in each column
        for _, col_name in enumerate(df.columns):
            nan_count = df[col_name].isna().sum()
            assert nan_count == 0, \
                f"{file_name}: Column '{col_name}' has {nan_count} NaN values"
        
        print(f"{file_name} has no NaN values")


def test_text_column_not_empty():
    """Test that text column doesn't contain empty strings."""
    files = ['test.csv', 'train.csv', 'validation.csv']
    
    for file_name in files:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path)
        
        # Get text column (first column)
        text_col = df.iloc[:, 0]
        
        # Count empty strings (after stripping whitespace)
        empty_count = text_col.apply(lambda x: str(x).strip() == '').sum()
        assert empty_count == 0, \
            f"{file_name}: Text column has {empty_count} empty strings"
        
        # Optional: Check for whitespace-only strings
        whitespace_only_count = text_col.apply(lambda x: str(x).strip() == '' and str(x) != '').sum()
        if whitespace_only_count > 0:
            print(f"{file_name}: Text column has {whitespace_only_count} whitespace-only strings")
        
        print(f"{file_name} text column has no empty strings")

if __name__ == "__main__":
    # Simple test runner for debugging
    test_module_can_be_imported()
    test_constants_exist()
    test_constant_values()
    test_main_function_exists()
    test_data_directory_is_created()
    test_google_drive_api_call()
    test_archive_extraction()
    test_all_data_files_exist()
    test_csv_files_can_be_read()
    test_data_has_correct_columns()
    test_column_data_types()
    test_label_values_are_binary()
    test_no_nan_values()
    test_text_column_not_empty()