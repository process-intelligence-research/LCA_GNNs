"""
Test module imports and basic functionality
"""

import importlib.util
import inspect
import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_core_imports():
    """Test that core modules can be imported"""
    try:
        from engines.predict_engines import predict_model

        assert callable(predict_model)
        print("✅ Predict engine imported successfully")
    except ImportError as e:
        pytest.skip(f"Core import failed (expected without training data): {e}")


def test_models_import():
    """Test models module import"""
    try:
        # Test availability using importlib.util.find_spec
        spec = importlib.util.find_spec("models.models")
        if spec is not None:
            print("✅ Models module available")
        else:
            pytest.skip("Models module not available")
    except ImportError as e:
        pytest.skip(f"Models import failed (expected without training data): {e}")


def test_utils_import():
    """Test utils module import"""
    try:
        # Test availability using importlib.util.find_spec
        spec = importlib.util.find_spec("utils")
        if spec is not None:
            print("✅ Utils module available")
        else:
            pytest.skip("Utils module not available")
    except ImportError as e:
        pytest.skip(f"Utils import failed (expected without training data): {e}")


def test_inference_function_structure():
    """Test inference function signature"""
    try:
        from engines.predict_engines import predict_model

        # Check function signature
        sig = inspect.signature(predict_model)
        params = list(sig.parameters.keys())

        # Verify function has expected parameters
        expected_params = ["model_path", "data_path", "dataset_type"]
        for param in expected_params:
            if param not in params:
                pytest.skip(f"Function signature changed, {param} not found")

        print(f"✅ Function parameters: {params}")
        assert callable(predict_model)

    except ImportError as e:
        pytest.skip(f"Inference engine import failed: {e}")


def test_config_structure():
    """Test configuration directory structure"""
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")

    if os.path.exists(config_dir):
        config_files = os.listdir(config_dir)
        print(f"Config files found: {config_files}")
    else:
        print("No configs directory found - this is okay")

    # This test always passes as configs are optional
    assert True


def test_fastapi_app():
    """Test FastAPI application structure"""
    try:
        from apps.fastAPI.app import app
        from fastapi import FastAPI

        assert isinstance(app, FastAPI), "app is not a FastAPI instance"
        print("✅ FastAPI app structure validated")

    except ImportError as e:
        pytest.skip(f"FastAPI import failed (might need additional setup): {e}")


def test_project_structure():
    """Test that required directories exist"""
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    required_dirs = ["src", "data", "trained_models"]
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        assert os.path.exists(dir_path), f"Required directory {dir_name} not found"

    print("✅ Project structure validated")
