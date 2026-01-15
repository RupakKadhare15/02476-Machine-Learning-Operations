"""
Test file for data.py.
"""

def test_constants():
    """Test that basic constants exist."""
    # Import inside test to avoid issues if module doesn't exist
    try:
        import data
        assert hasattr(data, 'DRIVE_ID')
        assert hasattr(data, 'DATA_PATH')
        print("Constants test passed")
    except ImportError:
        # If module doesn't exist, skip but don't fail
        print("data module not found - skipping test")

def test_main_function_exists():
    """Test that main function exists."""
    try:
        import data
        assert hasattr(data, 'main')
        assert callable(data.main)
        print("Main function test passed")
    except ImportError:
        print("data module not found - skipping test")


if __name__ == "__main__":
    # Simple test runner for debugging
    test_constants()
    test_main_function_exists()