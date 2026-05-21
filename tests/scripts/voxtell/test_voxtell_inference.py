import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Adjust the import according to your PYTHONPATH. Assumes the project root is executable.
import scripts.voxtell.voxtell_inference as voxtell_inference

@pytest.fixture
def mock_env_vars():
    """Defines required environment variables to bypass the initial validation block."""
    return {
        "MODEL_DIR": "/tmp/dummy_models",
        "DATA_PREP_DIR": "/tmp/dummy_prep",
        "DATA_PRED_DIR": "/tmp/dummy_pred",
        "DATASET_JSON": "/tmp/dummy_dataset.json",
        "CUDA_VISIBLE_DEVICES": "0",
        "DEFAULT_DEVICE": "cpu"
    }

@patch.dict(os.environ, {}, clear=True)
def test_missing_env_vars_raises_error():
    """Validates that the script aborts immediately if the .env file is incomplete."""
    with pytest.raises(ValueError, match="Error: Missing environment variables in .env"):
        voxtell_inference.main()

@patch.dict(os.environ, {}, clear=True)
@patch("scripts.voxtell.voxtell_inference.snapshot_download")
@patch("scripts.voxtell.voxtell_inference.VoxTellPredictor")
@patch("scripts.voxtell.voxtell_inference.nib")
@patch("builtins.open")
@patch("scripts.voxtell.voxtell_inference.os.path.exists")
def test_successful_inference_loop_and_4d_stacking(
    mock_exists, 
    mock_open, 
    mock_nib, 
    mock_predictor_class, 
    mock_snapshot,
    mock_env_vars
):
    """Validates the complete pipeline: model loading, vectorized inference, and 4D stacking."""
    # Inject test environment variables
    os.environ.update(mock_env_vars)
    
    # 1. Configure I/O Mocks
    mock_exists.return_value = True
    
    dummy_json_data = {
        "val": [
            {
                "name": "case_001.nii.gz",
                "findings": {
                    "0": "finding A",
                    "1": "finding B"
                }
            }
        ]
    }
    
    # Simulate reading dataset.json
    mock_file = MagicMock()
    mock_file.read.return_value = json.dumps(dummy_json_data)
    mock_open.return_value.__enter__.return_value = mock_file

    # 2. Configure NIfTI Reading Mocks (Nibabel)
    mock_nii_obj = MagicMock()
    dummy_img = np.zeros((32, 32, 32))  # Dummy input tensor in RAS orientation
    dummy_affine = np.eye(4)
    mock_nii_obj.get_fdata.return_value = dummy_img
    mock_nii_obj.affine = dummy_affine
    mock_nib.load.return_value = mock_nii_obj

    # 3. Configure Model Mocks (VoxTell)
    mock_predictor_instance = mock_predictor_class.return_value
    # For 2 findings, the model must return a tensor of shape (2, H, W, D)
    mock_predictor_instance.predict_single_image.return_value = np.ones((2, 32, 32, 32))

    # Execute script
    voxtell_inference.main()

    # Control flow assertions
    mock_snapshot.assert_called_once()
    mock_predictor_class.assert_called_once()
    mock_nib.load.assert_called_once_with("/tmp/dummy_prep/case_001_ct.nii.gz")
    
    # Check that prompts were passed in the exact JSON order
    mock_predictor_instance.predict_single_image.assert_called_once_with(
        dummy_img, 
        ["finding A", "finding B"]
    )

    # Validate assembly and saving of the 4D tensor
    mock_nib.Nifti1Image.assert_called_once()
    args, _ = mock_nib.Nifti1Image.call_args
    output_tensor, output_affine = args[0], args[1]
    
    assert output_tensor.shape == (2, 32, 32, 32), "The output tensor does not have the expected 4D shape (F, H, W, D)."
    assert output_tensor.dtype == np.uint8, "Cast to uint8 failed, risking OOM or I/O bottleneck."
    np.testing.assert_array_equal(output_affine, dummy_affine), "The affine matrix was not preserved during saving."
    
    mock_nib.save.assert_called_once()
    assert mock_nib.save.call_args[0][1] == "/tmp/dummy_pred/case_001.nii.gz"