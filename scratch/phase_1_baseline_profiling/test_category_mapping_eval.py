import os
import re
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

# Dictionary mapping common report descriptions to standard VoxTell pre-training entity queries
ENTITY_MAPPING_RULES = [
    (r'\b(nodule|nodules|mass|masses)\b', 'pulmonary nodule'),
    (r'\b(ground[- ]glass|ggo)\b', 'ground glass opacity'),
    (r'\b(pleural effusion|effusion)\b', 'pleural effusion'),
    (r'\b(atelectasis|atelectatic)\b', 'atelectasis'),
    (r'\b(consolidation|consolidative)\b', 'consolidation'),
    (r'\b(bronchial wall thickening|thickening of bronchial)\b', 'bronchial wall thickening'),
    (r'\b(bronchiectasis|bronchiectatic)\b', 'bronchiectasis'),
    (r'\b(emphysema|emphysematous)\b', 'emphysema'),
    (r'\b(pneumothorax)\b', 'pneumothorax'),
    (r'\b(pleural thickening)\b', 'pleural thickening'),
    (r'\b(honeycombing)\b', 'honeycombing'),
    (r'\b(septal thickening)\b', 'septal thickening'),
    (r'\b(opacity|opacities)\b', 'focal opacity'),
    (r'\b(cyst|cysts)\b', 'cyst'),
    (r'\b(scarring|fibrosis|fibrotic)\b', 'scarring'),
]

def extract_canonical_entity(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for pattern, canonical in ENTITY_MAPPING_RULES:
        if re.search(pattern, prompt_lower):
            return canonical
    # Fallback: strip adjectives & measurements
    cleaned = re.sub(r',?\s*measuring\s+\d+.*', '', prompt, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(stable|nonspecific|minimal|mild|moderate|severe|focal)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else prompt

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    load_dotenv(override=True)
    img_dir = os.environ["IMG_RAW_DIR"]
    gt_dir = os.environ["SEG_RAW_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = "/home/jdeferrari/rex_project/models/voxtell_v1.1"

    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])[50:60]  # Test on 10 cases from the new split
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5
    reader = NibabelIOWithReorient()
    
    orig_dices = []
    canon_dices = []
    
    print(f"\n==========================================")
    print(f"=== TESTING CANONICAL ENTITY QUERIES ON 10 SCANS ===")
    print(f"==========================================")
    
    for entry in val_cases:
        scan_id = entry["name"].replace(".nii.gz", "")
        nifti_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path): continue
        
        img, img_props = reader.read_images([nifti_path])
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata(dtype=np.float32)
        if gt_data.ndim == 3: gt_data = np.expand_dims(gt_data, axis=-1)
        if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data_f = np.moveaxis(gt_data, -1, 0)
        else:
            gt_data_f = gt_data
            
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            orig_prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            orig_prompts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
            
        canon_prompts = [extract_canonical_entity(p) for p in orig_prompts]
        
        def run_reorient(prompts_list):
            with torch.no_grad():
                raw_seg = predictor.predict_single_image(img, prompts_list)
            pred_xyzf = np.transpose(raw_seg, (3, 2, 1, 0))
            reoriented_affine = img_props['nibabel_stuff']['reoriented_affine']
            pred_nib = nib.Nifti1Image(pred_xyzf, reoriented_affine)
            original_affine = img_props['nibabel_stuff']['original_affine']
            img_ornt = io_orientation(original_affine)
            ras_ornt = axcodes2ornt("RAS")
            from_canonical = ornt_transform(ras_ornt, img_ornt)
            pred_nib_back = pred_nib.as_reoriented(from_canonical)
            pred_back_data = np.asanyarray(pred_nib_back.dataobj).astype(np.uint8)
            return np.transpose(pred_back_data, (3, 0, 1, 2))

        pred_orig = run_reorient(orig_prompts)
        pred_canon = run_reorient(canon_prompts)
        
        print(f"\n--- SCAN {scan_id} ---")
        for f_idx in range(gt_data_f.shape[0]):
            gt_m = gt_data_f[f_idx]
            d_o = compute_dice(pred_orig[f_idx], gt_m)
            d_c = compute_dice(pred_canon[f_idx], gt_m)
            orig_dices.append(d_o)
            canon_dices.append(d_c)
            print(f"  Finding #{f_idx} (GT vox: {int(np.sum(gt_m>0))})")
            print(f"    Orig: '{orig_prompts[f_idx][:35]}...' -> Dice: {d_o:.4f}")
            print(f"    Canon: '{canon_prompts[f_idx]}' -> Dice: {d_c:.4f}")

    print("\n==========================================")
    print("=== SUMMARY RESULTS OVER 10 SCANS ===")
    print(f"Original Prompts Average Dice: {np.mean(orig_dices):.4f}")
    print(f"Canonical Entity Queries Average Dice: {np.mean(canon_dices):.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()
