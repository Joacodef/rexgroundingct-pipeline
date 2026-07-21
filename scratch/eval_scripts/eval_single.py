import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
import sys

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

load_dotenv(override=True)
gt_dir = os.environ["SEG_RAW_DIR"]
pred_dir = "/tmp/voxtell_0.5_test"
scan_id = "train_18379_a_1"

gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")

if not os.path.exists(pred_path):
    print(f"Prediction not found: {pred_path}")
    sys.exit(1)

gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
pred_img = nib.load(pred_path).get_fdata(dtype=np.float32)

if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
    gt_img = np.moveaxis(gt_img, -1, 0)

dices = []
for f_idx in range(gt_img.shape[0]):
    gt_mask = gt_img[f_idx]
    pred_mask = pred_img[f_idx]
    dice = compute_dice(pred_mask, gt_mask)
    dices.append(dice)
    print(f"Finding {f_idx}: Dice {dice:.4f}, Voxel Count: {np.sum(gt_mask > 0)}")

print(f"\nAverage Dice: {np.mean(dices):.4f}")
print(f"Hit Rate: {np.mean([1 if d >= 0.1 else 0 for d in dices]):.2%}")
