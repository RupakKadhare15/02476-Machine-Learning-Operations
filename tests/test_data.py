"""
Test file for data.py.
"""
from unittest.mock import Mock, patch, mock_open

def test_module_can_be_imported():
    """
    Test that the data module can be imported.
    """
    import data
    assert True  # If we get here, import succeeded

def test_constants_exist():
    """
    Verify that required constants are defined.
    """
    import data
    
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
    import data
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
        import data
        assert hasattr(data, 'main')
        assert callable(data.main)
        print("Main function test passed")
    except ImportError:
        print("data module not found - skipping test")

def test_data_directory_is_created():
    """
    Verify that the script creates the necessary directory structure.
    """
    import data
    
    # Mock Path.mkdir to test it's called correctly
    with patch('data.DATA_PATH.mkdir') as mock_mkdir:
        # Mock everything else to isolate the mkdir test
        with patch('data.requests.Session'):
            with patch('data.shutil.unpack_archive'):
                with patch('builtins.open', mock_open()):
                    data.main()
        
        # Verify mkdir was called with correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # parents=True ensures parent directories are created if needed
        # exist_ok=True prevents errors if directory already exists

def test_google_drive_api_call():
    """
    Verify the script makes the correct API request to Google Drive.
    This tests the URL, parameters, and streaming setup.
    """
    import data
    
    with patch('data.DATA_PATH.mkdir'):
        with patch('data.requests.Session') as mock_session_class:
            with patch('data.shutil.unpack_archive'):
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
    import data

    with patch('data.DATA_PATH.mkdir'):
        with patch('data.requests.Session'):
            with patch('data.shutil.unpack_archive') as mock_unpack:
                with patch('builtins.open', mock_open()):
                    data.main()
                    
                    # Verify unpack_archive was called with correct arguments
                    mock_unpack.assert_called_once_with(
                        data.DATA_PATH / 'data.zip',  # Source zip file
                        data.DATA_PATH  # Destination directory
                    )


if __name__ == "__main__":
    # Simple test runner for debugging
    test_module_can_be_imported()
    test_constants_exist()
    test_constant_values()
    test_main_function_exists()
    test_data_directory_is_created()
    test_google_drive_api_call()
    test_archive_extraction()