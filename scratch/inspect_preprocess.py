"""
VoxTell Batch Zero-Shot Inference Pipeline for ReXGroundingCT.

This script performs batch inference using the VoxTell model on preprocessed 
3D CT scans, generating 4D segmentation masks (F, H, W, D) guided by free-text 
prompts. It ensures strict positional alignment with the ground truth JSON 
to maintain compatibility with the official `rexrank_eval.py` script.

Input/Output Contract:
- Inputs: Configuration relies exclusively on environment variables loaded via 
  `.env` (MODEL_DIR, DATA_PREP_DIR, DATA_PRED_DIR, DATASET_JSON). Optional 
  `--split` CLI argument is available to target specific dataset partitions.
- Outputs: 4D NIfTI files saved in `DATA_PRED_DIR` preserving the original affine matrix.
"""

import os
import json
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# 1. Strictly isolate GPU (Node policy) MUST happen before VoxTell imports
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Import VoxTell dependencies after setting environment variables
from voxtell.inference.predictor import VoxTellPredictor

def main():
    # Parse CLI arguments for split selection
    parser = argparse.ArgumentParser(description="VoxTell Batch Zero-Shot Inference")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate (train, val, test)")
    args = parser.parse_args()

    # Inject paths from .env file
    download_dir = os.getenv("MODEL_DIR")
    
    # Prefer fast volatile storage (/tmp) if available, fallback to persistent storage
    data_prep_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    output_dir = os.getenv("TMP_PRED_DIR") or os.getenv("DATA_PRED_DIR")
    dataset_json = os.getenv("DATASET_JSON")

    # Security validation for critical environment variables
    if not all([download_dir, data_prep_dir, output_dir, dataset_json]):
        raise ValueError("Error: Missing environment variables in .env (MODEL_DIR, DATA_PREP_DIR/TMP_PREP_DIR, DATA_PRED_DIR/TMP_PRED_DIR, DATASET_JSON).")

    # Prepare directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Device Configuration
    device = torch.device(os.getenv("DEFAULT_DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    
    # We will download the paper version (v1.0)
    model_name = "voxtell_v1.0" 
    
    # Get Weights from Hugging Face
    snapshot_download(
        repo_id="mrokuss/VoxTell", 
        allow_patterns=[f"{model_name}/*", "*.json"], 
        local_dir=download_dir
    )
    voxtell_weights_dir = os.path.join(download_dir, model_name)

    # Initialize Predictor
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)

    # Load Ground Truth metadata
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get(args.split, [])
    if not entries:
        print(f"[WARNING] No cases found for split '{args.split}'. Check dataset.json structure.")
        return

    missing_files_count = 0

    # Batch Inference Loop
    for entry in tqdm(entries, desc=f"Evaluating {args.split} Scans"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue
            
        nifti_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
        
        if not os.path.exists(nifti_path):
            tqdm.write(f"[WARNING] Preprocessed file not found: {nifti_path}. Skipping.")
            missing_files_count += 1
            continue
            
        # Load NIfTI directly using standard Nibabel to keep the standardized RAS orientation
        nii_obj = nib.load(nifti_path)
        img = nii_obj.get_fdata(dtype=np.float32)
        affine = nii_obj.affine
        
        findings = entry.get('findings', {})
        if not findings:
            continue
            
        # Extract prompts, handling both dictionary and list formats (for testing compatibility)
        if isinstance(findings, dict):
            # Sort keys numerically to ensure strict alignment with the channel order
            sorted_keys = sorted(findings.keys(), key=int)
            text_prompts = []
            for k in sorted_keys:
                val = findings[k]
                if isinstance(val, dict):
                    text_prompts.append(val.get('text', ''))
                else:
                    text_prompts.append(str(val))
        else:
            text_prompts = [f['text'] if isinstance(f, dict) else f for f in findings]
        
        # Inference
        with torch.no_grad():
            # Input image is passed directly in RAS orientation (X, Y, Z)
            # Output: (num_prompts, X, Y, Z) in the same RAS space, matching the GT exactly
            pred_4d = predictor.predict_single_image(img, text_prompts)
            
        # Cast to uint8 to minimize memory footprint and I/O bottleneck
        pred_4d = pred_4d.astype(np.uint8)
        
        # Save as NIfTI preserving the preprocessed space affine
        out_nii = nib.Nifti1Image(pred_4d, affine)
        out_path = os.path.join(output_dir, f"{scan_id}.nii.gz")
        nib.save(out_nii, out_path)

    if missing_files_count > 0:
        print(f"\n[INFO] Inference completed. Skipped {missing_files_count} missing files. Make sure to run the preprocessing script for the '{args.split}' split.")

if __name__ == "__main__":
    main()