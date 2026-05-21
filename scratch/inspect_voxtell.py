import os
import sys
import json
import torch
import numpy as np
import inspect
import nibabel as nib
from dotenv import load_dotenv

# Import VoxTell predictor
try:
    from voxtell.inference.predictor import VoxTellPredictor
except ImportError as e:
    os.makedirs("logs", exist_ok=True)
    with open("logs/inspect_voxtell.log", "w", encoding="utf-8") as f:
        f.write(f"[ERROR] Failed to import VoxTell: {e}\n")
    print(f"[ERROR] Failed to import VoxTell: {e}")
    exit(1)

def main():
    load_dotenv(override=True)
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/inspect_voxtell.log"
    
    # Redirect print to both console and a log file
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    
    print("=" * 70)
    print("      INSPECTOR: VOXTELL METHOD & RUNTIME OUTPUT")
    print("=" * 70)
    
    # 1. Print the source code of predict_single_image to understand its behavior
    print("\n--- [SOURCE CODE] VoxTellPredictor.predict_single_image ---")
    try:
        source = inspect.getsource(VoxTellPredictor.predict_single_image)
        print(source)
    except Exception as e:
        print(f"Failed to get source code: {e}")
        
    print("-" * 70)

    # 2. Run on the first case and print the raw, pre-caste prediction details
    data_prep_dir = os.getenv("TMP_PREP_DIR") or os.getenv("DATA_PREP_DIR")
    dataset_json = os.getenv("DATASET_JSON")
    
    if not all([data_prep_dir, dataset_json]):
        print("[WARNING] Missing path config, skipping runtime prediction test.")
        return

    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
    
    entries = metadata.get("val", [])
    if not entries:
        print("[WARNING] No validation cases found.")
        return
        
    entry = entries[0]
    scan_id = entry["name"].replace(".nii.gz", "")
    nifti_path = os.path.join(data_prep_dir, f"{scan_id}_ct.nii.gz")
    
    if not os.path.exists(nifti_path):
        print(f"[WARNING] Preprocessed file not found: {nifti_path}. Cannot run prediction test.")
        return

    print(f"\n--- [RUNTIME TEST] Running VoxTell on {scan_id} ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    download_dir = os.getenv("MODEL_DIR")
    voxtell_weights_dir = os.path.join(download_dir, "voxtell_v1.0")
    
    print(f"Loading model from {voxtell_weights_dir}...")
    try:
        predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
        
        # Standard nibabel loader to preserve RAS orientation and shape
        nii_obj = nib.load(nifti_path)
        img = nii_obj.get_fdata(dtype=np.float32)
        
        findings = entry.get('findings', {})
        if isinstance(findings, dict):
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
        
        print(f"Prompts: {text_prompts}")
        print(f"Input shape: {img.shape}")
        print("Running predictor.predict_single_image...")
        
        with torch.no_grad():
            voxtell_seg = predictor.predict_single_image(img, text_prompts)
            
        print("\n--- [RESULTS] Raw Output properties ---")
        print(f"Type             : {type(voxtell_seg)}")
        print(f"Dtype            : {voxtell_seg.dtype if hasattr(voxtell_seg, 'dtype') else 'N/A'}")
        print(f"Shape            : {voxtell_seg.shape}")
        if hasattr(voxtell_seg, 'min'):
            print(f"Min value        : {voxtell_seg.min()}")
            print(f"Max value        : {voxtell_seg.max()}")
            print(f"Mean value       : {voxtell_seg.mean()}")
            unique_vals = np.unique(voxtell_seg)
            print(f"Unique values count: {len(unique_vals)}")
            print(f"Unique values (up to 10): {unique_vals.tolist()[:10]}")
            
    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
