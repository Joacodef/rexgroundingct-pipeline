import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

load_dotenv(override=True)
gt_dir = os.environ["SEG_RAW_DIR"]
pred_dir = os.environ["DATA_PRED_DIR"]
dataset_json = os.environ["DATASET_JSON"]

def main():
    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    
    def analyze_subset(cases, name):
        gt_sizes = []
        pred_sizes = []
        empty_preds = 0
        prompt_lens = []
        findings_count = 0
        
        for entry in cases:
            scan_id = entry["name"].replace(".nii.gz", "")
            gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
            pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
            
            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                continue
                
            gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
            pred_img = nib.load(pred_path).get_fdata(dtype=np.float32)
            
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
                
            for f_idx in range(gt_img.shape[0]):
                findings_count += 1
                gt_vox = np.sum(gt_img[f_idx] > 0)
                pred_vox = np.sum(pred_img[f_idx] > 0)
                gt_sizes.append(gt_vox)
                pred_sizes.append(pred_vox)
                if pred_vox == 0:
                    empty_preds += 1
                if f_idx < len(texts):
                    prompt_lens.append(len(texts[f_idx].split()))
                    
        print(f"\n==========================================")
        print(f"=== {name} ANALYSIS ===")
        print(f"==========================================")
        print(f"Total scans: {len(cases)} | Total findings: {findings_count}")
        print(f"Empty Predictions (0 voxels): {empty_preds} / {findings_count} ({empty_preds/findings_count*100:.1f}%)")
        print(f"Mean GT size (voxels): {np.mean(gt_sizes):.1f} | Median GT size: {np.median(gt_sizes):.1f}")
        print(f"Mean Pred size (voxels): {np.mean(pred_sizes):.1f} | Median Pred size: {np.median(pred_sizes):.1f}")
        print(f"Small findings (<1k voxels): {sum(1 for s in gt_sizes if s < 1000)} ({sum(1 for s in gt_sizes if s < 1000)/findings_count*100:.1f}%)")
        print(f"Medium findings (1k-10k voxels): {sum(1 for s in gt_sizes if 1000 <= s < 10000)} ({sum(1 for s in gt_sizes if 1000 <= s < 10000)/findings_count*100:.1f}%)")
        print(f"Large findings (>10k voxels): {sum(1 for s in gt_sizes if s >= 10000)} ({sum(1 for s in gt_sizes if s >= 10000)/findings_count*100:.1f}%)")
        print(f"Mean Prompt length (words): {np.mean(prompt_lens):.1f}")

    analyze_subset(val_cases[:50], "FIRST 50 (PAPER VAL)")
    analyze_subset(val_cases[50:], "NEXT 150 (NEW MICCAI VAL)")

if __name__ == "__main__":
    main()
