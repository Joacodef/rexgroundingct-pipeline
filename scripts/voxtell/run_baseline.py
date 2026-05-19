import os
import json
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# Strictly isolate GPU 0 before importing PyTorch/MONAI
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def main():
    # 1. Define parser with .env variables as defaults
    parser = argparse.ArgumentParser(description="Baseline Batch Inference")
    
    # Priority: Command line args > TMP_PREP_DIR from .env > DATA_PREP_DIR from .env
    default_input = os.getenv("TMP_PREP_DIR", os.getenv("DATA_PREP_DIR"))
    default_output = os.getenv("DATA_PRED_DIR", "/tmp/predictions")
    
    parser.add_argument("--input_dir", type=str, default=default_input, help="Path to preprocessed CTs")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Output path for generated masks")
    
    args = parser.parse_args()

    # 2. Extract paths
    data_preprocessed_dir = args.input_dir
    out_dir = args.output_dir
    json_path = os.getenv("DATASET_JSON")
    model_dir = os.getenv("MODEL_DIR")

    # Validate critical paths
    if not all([data_preprocessed_dir, json_path, out_dir, model_dir]):
        raise ValueError("Error: Missing critical paths. Check your .env variables.")

    os.makedirs(out_dir, exist_ok=True)

    # 3. Load dataset.json
    print(f"Loading dataset metadata from: {json_path}")
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Extract test split 
    test_entries = dataset_info.get("test", [])
    print(f"Total evaluation cases found: {len(test_entries)}")
    print(f"Target input directory: {data_preprocessed_dir}")

    # 4. Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading VoxTell weights into VRAM...")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    reader = NibabelIOWithReorient()

    # 5. Inference loop
    for entry in tqdm(test_entries, desc="Zero-Shot Inference"):
        prompts = entry.get("findings", []) 
        
        # Extract base name without extension
        base_name = entry["name"].replace(".nii.gz", "")
        
        # Construct physical filename for preprocessed CT matching Jumbito's structure
        img_filename = f"{base_name}_ct.nii.gz"
        img_path = os.path.join(data_preprocessed_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Skipping {img_filename}, not found in {img_path}")
            continue

        # Inference
        img, _ = reader.read_images([img_path])
        pred_tensor = predictor.predict_single_image(img, prompts)
        
        # Retrieve original affine matrix for spatial alignment
        original_nifti = nib.load(img_path)
        affine = original_nifti.affine
        
        # Force np.uint8 to save memory, as required by rexrank_eval.py
        pred_nifti = nib.Nifti1Image(pred_tensor.astype(np.uint8), affine)
        
        # Save using the exact original JSON filename for evaluation compatibility
        out_path = os.path.join(out_dir, entry["name"]) 
        nib.save(pred_nifti, out_path)

    print(f"Process finished. Predictions saved to: {out_dir}")

if __name__ == "__main__":
    main()