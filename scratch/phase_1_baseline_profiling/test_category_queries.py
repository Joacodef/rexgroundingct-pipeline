import os
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

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

    scan_id = "train_18382_b_2"
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
        
    category_prompts = [
        "pulmonary nodule",
        "ground glass opacity"
    ]
    
    device = torch.device("cuda:0")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    predictor.tile_step_size = 0.5
    
    data_prep, bbox, orig_shape = predictor.preprocess(img)
    emb_cat = predictor.embed_text_prompts(category_prompts)
    
    with torch.no_grad():
        logits_cat = predictor.predict_sliding_window_return_logits(data_prep, emb_cat).to("cpu")
        probs_cat = torch.sigmoid(logits_cat.float()).numpy()
        
    def reorient_mask(raw_seg):
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

    binary_seg = (probs_cat > 0.5).astype(np.uint8)
    reoriented_pred = reorient_mask(binary_seg)
    
    print(f"\n==========================================")
    print(f"=== TESTING SHORT CATEGORY QUERIES: {scan_id} ===")
    print(f"==========================================")
    
    for f_idx, cat_text in enumerate(category_prompts):
        p_map = probs_cat[f_idx]
        gt_m = gt_data_f[f_idx]
        pr_m = reoriented_pred[f_idx]
        
        gt_vox = int(np.sum(gt_m > 0))
        pr_vox = int(np.sum(pr_m > 0))
        d = compute_dice(pr_m, gt_m)
        
        print(f"\nFinding #{f_idx} (GT Voxels: {gt_vox}):")
        print(f"  Category Query: '{cat_text}'")
        print(f"    Max prob: {p_map.max():.4f} | 99.9th percentile: {np.percentile(p_map, 99.9):.4f}")
        print(f"    Pred Voxels (>0.5): {pr_vox} | Dice: {d:.4f}")

if __name__ == "__main__":
    main()
