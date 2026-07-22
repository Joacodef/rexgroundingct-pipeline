import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor

load_dotenv(override=True)
gt_dir = os.environ["SEG_RAW_DIR"]
pred_dir = os.environ["DATA_PRED_DIR"]
dataset_json = os.environ["DATASET_JSON"]

def analyze_entry(entry):
    scan_id = entry["name"].replace(".nii.gz", "")
    gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
    
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        return None
        
    gt_img = np.asanyarray(nib.load(gt_path).dataobj)
    pred_img = np.asanyarray(nib.load(pred_path).dataobj)
    
    if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
    if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
        gt_img = np.moveaxis(gt_img, -1, 0)
        
    if pred_img.ndim == 3: pred_img = np.expand_dims(pred_img, axis=-1)
    if pred_img.ndim == 4 and pred_img.shape[-1] < np.min(pred_img.shape[:-1]):
        pred_img = np.moveaxis(pred_img, -1, 0)
        
    findings = entry.get("findings", {})
    if isinstance(findings, dict):
        texts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted(findings.keys(), key=int)]
    else:
        texts = [f["text"] if isinstance(f, dict) else str(f) for f in findings]
        
    gt_sizes = []
    pred_sizes = []
    empty_preds = 0
    prompt_lens = []
    
    for f_idx in range(gt_img.shape[0]):
        gt_vox = int(np.sum(gt_img[f_idx] > 0))
        pred_vox = int(np.sum(pred_img[f_idx] > 0))
        gt_sizes.append(gt_vox)
        pred_sizes.append(pred_vox)
        if pred_vox == 0:
            empty_preds += 1
        if f_idx < len(texts):
            prompt_lens.append(len(texts[f_idx].split()))
            
    return {
        "gt_sizes": gt_sizes,
        "pred_sizes": pred_sizes,
        "empty_preds": empty_preds,
        "prompt_lens": prompt_lens,
        "num_findings": gt_img.shape[0]
    }

def main():
    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    def run_subset(cases, name):
        with ProcessPoolExecutor(max_workers=16) as executor:
            res = list(executor.map(analyze_entry, cases))
        res = [r for r in res if r is not None]
        
        all_gt = [s for r in res for s in r["gt_sizes"]]
        all_pred = [s for r in res for s in r["pred_sizes"]]
        all_empty = sum(r["empty_preds"] for r in res)
        all_prompts = [l for r in res for l in r["prompt_lens"]]
        total_findings = sum(r["num_findings"] for r in res)
        
        print(f"\n==========================================")
        print(f"=== {name} ANALYSIS ===")
        print(f"==========================================")
        print(f"Total scans: {len(cases)} | Total findings: {total_findings}")
        print(f"Empty Predictions (0 voxels): {all_empty} / {total_findings} ({all_empty/total_findings*100:.1f}%)")
        print(f"Mean GT size (voxels): {np.mean(all_gt):.1f} | Median GT size: {np.median(all_gt):.1f}")
        print(f"Mean Pred size (voxels): {np.mean(all_pred):.1f} | Median Pred size: {np.median(all_pred):.1f}")
        print(f"Small findings (<1k voxels): {sum(1 for s in all_gt if s < 1000)} ({sum(1 for s in all_gt if s < 1000)/total_findings*100:.1f}%)")
        print(f"Medium findings (1k-10k voxels): {sum(1 for s in all_gt if 1000 <= s < 10000)} ({sum(1 for s in all_gt if 1000 <= s < 10000)/total_findings*100:.1f}%)")
        print(f"Large findings (>10k voxels): {sum(1 for s in all_gt if s >= 10000)} ({sum(1 for s in all_gt if s >= 10000)/total_findings*100:.1f}%)")
        print(f"Mean Prompt length (words): {np.mean(all_prompts):.1f}")

    run_subset(val_cases[:50], "FIRST 50 (PAPER VAL)")
    run_subset(val_cases[50:], "NEXT 150 (NEW MICCAI VAL)")

if __name__ == "__main__":
    main()
