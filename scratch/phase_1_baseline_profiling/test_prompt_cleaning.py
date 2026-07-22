import os
import sys
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scripts.voxtell.prompt_normalizer import clean_finding_prompt



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
        
    val_cases = meta.get("val", [])
    scan_id = "train_18382_b_2"
    entry = next(e for e in val_cases if e["name"].replace(".nii.gz", "") == scan_id)
    
    nifti_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
    gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
    
    reader = NibabelIOWithReorient()
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
        original_prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
    else:
        original_prompts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
        
    cleaned_prompts = [clean_finding_prompt(p) for p in original_prompts]
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5  # Full precision tile step
    
    def run_inference_and_reorient(prompts_list):
        with torch.no_grad():
            voxtell_seg = predictor.predict_single_image(img, prompts_list)
        pred_xyzf = np.transpose(voxtell_seg, (3, 2, 1, 0))
        reoriented_affine = img_props['nibabel_stuff']['reoriented_affine']
        pred_nib = nib.Nifti1Image(pred_xyzf, reoriented_affine)
        
        original_affine = img_props['nibabel_stuff']['original_affine']
        img_ornt = io_orientation(original_affine)
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        
        pred_nib_back = pred_nib.as_reoriented(from_canonical)
        pred_back_data = np.asanyarray(pred_nib_back.dataobj).astype(np.uint8)
        return np.transpose(pred_back_data, (3, 0, 1, 2))

    print(f"\n==========================================")
    print(f"=== PROMPT CLEANING BENCHMARK ON {scan_id} ===")
    print(f"==========================================")
    
    print("\n1. Running inference with ORIGINAL prompts...")
    pred_orig = run_inference_and_reorient(original_prompts)
    
    print("\n2. Running inference with AUTOMATICALLY CLEANED prompts...")
    pred_clean = run_inference_and_reorient(cleaned_prompts)
    
    for f_idx in range(gt_data_f.shape[0]):
        gt_m = gt_data_f[f_idx]
        gt_vox = int(np.sum(gt_m > 0))
        
        pr_orig_m = pred_orig[f_idx]
        pr_orig_vox = int(np.sum(pr_orig_m > 0))
        d_orig = compute_dice(pr_orig_m, gt_m)
        
        pr_clean_m = pred_clean[f_idx]
        pr_clean_vox = int(np.sum(pr_clean_m > 0))
        d_clean = compute_dice(pr_clean_m, gt_m)
        
        print(f"\nFinding #{f_idx} (GT Voxels: {gt_vox}):")
        print(f"  ORIGINAL: '{original_prompts[f_idx]}'")
        print(f"            -> Pred Vox: {pr_orig_vox} | Dice: {d_orig:.4f}")
        print(f"  CLEANED : '{cleaned_prompts[f_idx]}'")
        print(f"            -> Pred Vox: {pr_clean_vox} | Dice: {d_clean:.4f}")

if __name__ == "__main__":
    main()
