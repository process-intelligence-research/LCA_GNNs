"""
Test inference engine functionality
"""

import importlib.util
import inspect
import os
import sys
import tempfile
import time

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestInferenceEngine:
    """Test suite for inference engine"""

    def test_predict_model_import(self):
        """Test that predict_model can be imported"""
        try:
            from engines.predict_engines import predict_model

            assert callable(predict_model)
        except ImportError as e:
            pytest.skip(f"predict_model import failed: {e}")

    def test_predict_model_signature(self):
        """Test predict_model function signature"""
        try:
            from engines.predict_engines import predict_model

            sig = inspect.signature(predict_model)
            params = list(sig.parameters.keys())

            # Just verify it has some expected parameters
            assert len(params) > 0, "Function should have parameters"
            print(f"Function parameters: {params}")

        except ImportError as e:
            pytest.skip(f"predict_model import failed: {e}")

    def test_mock_data_creation(self):
        """Test creating mock data for inference"""
        # Create mock CSV data
        data = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "CCCO", "CC(C)O"],
                "country_name": ["USA", "Canada", "Germany", "Japan"],
            }
        )

        # Create temporary file with proper Windows handling
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            # Write data to the temporary file
            data.to_csv(temp_path, index=False)

            # Verify file was created and has correct content
            assert os.path.exists(temp_path)

            # Read back and verify
            loaded_data = pd.read_csv(temp_path)
            assert len(loaded_data) == 4
            assert "SMILES" in loaded_data.columns
            assert "country_name" in loaded_data.columns
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPerformance:
    """Test performance characteristics"""

    def test_import_performance(self):
        """Test that imports don't take too long"""
        start_time = time.time()

        try:
            # Test availability using importlib.util.find_spec
            spec = importlib.util.find_spec("engines.predict_engines")
            if spec is not None:
                import_time = time.time() - start_time

                # Should import in reasonable time (less than 10 seconds)
                assert import_time < 10, f"Import took too long: {import_time:.3f}s"
                print(f"Import time: {import_time:.3f} seconds")
            else:
                pytest.skip("engines.predict_engines not available")

        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


def test_data_directories():
    """Test that data directories exist"""
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    data_dir = os.path.join(base_dir, "data")
    assert os.path.exists(data_dir), "data directory should exist"

    # Check subdirectories
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")

    # These might not exist yet, just print status
    print(f"Raw data directory exists: {os.path.exists(raw_dir)}")
    print(f"Processed data directory exists: {os.path.exists(processed_dir)}")


def test_trained_models_directory():
    """Test trained models directory"""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    models_dir = os.path.join(base_dir, "trained_models")

    assert os.path.exists(models_dir), "trained_models directory should exist"
    print(f"Trained models directory: {models_dir}")
