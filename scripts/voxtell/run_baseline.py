import os
import json
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Strictly isolate GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def main():
    # Base paths injected from .env
    base_dir = os.getenv("BASE_DIR")
    data_preprocessed_dir = os.getenv("DATA_PREP_DIR")
    json_path = os.getenv("DATASET_JSON")
    out_dir = os.getenv("DATA_PRED_DIR")
    
    # Small security validation in case .env doesn't load properly
    if not all([base_dir, data_preprocessed_dir, json_path, out_dir]):
        raise ValueError("Error: Missing environment variables. Check your .env file.")

    os.makedirs(out_dir, exist_ok=True)

    # 2. Load dataset.json to extract prompts and filenames
    print(f"Loading dataset metadata from: {json_path}")
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Extract test split (official evaluator uses "test" key)
    test_entries = dataset_info.get("test", [])
    print(f"Total evaluation cases found: {len(test_entries)}")

    # 3. Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = os.getenv("MODEL_DIR")
    print("Loading VoxTell weights into VRAM...")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    reader = NibabelIOWithReorient()

    # 4. Inference loop
    for entry in tqdm(test_entries, desc="Zero-Shot Inference"):
        # NOTE: Check if text key in your dataset.json is "findings", "prompts" or "texts"
        # and adjust if necessary.
        prompts = entry.get("findings", []) 
        
        # Extract only filename to search in preprocessed folder
        img_filename = os.path.basename(entry["image_path"]) 
        seg_filename = os.path.basename(entry["seg_path"])
        
        img_path = os.path.join(data_preprocessed_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Skipping {img_filename}, not found.")
            continue

        # Inference
        img, _ = reader.read_images([img_path])
        pred_tensor = predictor.predict_single_image(img, prompts)
        
        # 5. Save 4D NIfTI
        # Steal 'affine' from preprocessed CT to ensure perfect spatial alignment
        original_nifti = nib.load(img_path)
        affine = original_nifti.affine
        
        # Force np.uint8 as required by rexrank_eval.py to save memory
        pred_nifti = nib.Nifti1Image(pred_tensor.astype(np.uint8), affine)
        out_path = os.path.join(out_dir, seg_filename) 
        nib.save(pred_nifti, out_path)

    print(f"Process finished. Predictions saved to: {out_dir}")

if __name__ == "__main__":
    main()
