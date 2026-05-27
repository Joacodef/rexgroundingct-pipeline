import os
import json
import torch
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from voxtell.inference.predictor import VoxTellPredictor

# Define lightweight ValidationPredictor to avoid loading heavy Qwen model
class DiagnosticPredictor(VoxTellPredictor):
    def __init__(self, model_dir, device, network):
        self.device = device
        self.normalization = ZScoreNormalization(intensityproperties={})
        self.tile_step_size = 0.5
        self.perform_everything_on_device = False
        self.max_text_length = 8192
        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        plans = load_json(join(model_dir, 'plans.json'))
        self.patch_size = plans['configurations']['3d_fullres']['patch_size']
        self.network = network

def load_voxtell_network(model_dir, device):
    from voxtell.model.voxtell_model import VoxTellModel
    import pydoc
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    
    plans = load_json(join(model_dir, 'plans.json'))
    arch_kwargs = plans['configurations']['3d_fullres']['architecture']['arch_kwargs']
    arch_kwargs = dict(**arch_kwargs)
    for required_import_key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
        if arch_kwargs[required_import_key] is not None:
            arch_kwargs[required_import_key] = pydoc.locate(arch_kwargs[required_import_key])
            
    model = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False
    )
    
    checkpoint = torch.load(
        join(model_dir, 'fold_0', 'checkpoint_final.pth'),
        map_location=torch.device('cpu'),
        weights_only=False
    )
    model.load_state_dict(checkpoint['network_weights'])
    model = model.to(device)
    return model

def main():
    load_dotenv(override=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset_json_path = os.getenv("DATASET_JSON", "data/dataset.json")
    img_dir = os.getenv("IMG_RAW_DIR")
    seg_dir = os.getenv("SEG_RAW_DIR")
    
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Pick the first case from validation split
    val_entry = dataset['val'][0]
    scan_id = val_entry['name'].replace('.nii.gz', '')
    findings = val_entry['findings']
    
    print(f"====================================================")
    # Print findings
    findings_list = []
    if isinstance(findings, dict):
        sorted_keys = sorted(findings.keys(), key=int)
        for k in sorted_keys:
            findings_list.append(findings[k]['text'] if isinstance(findings[k], dict) else findings[k])
    else:
        findings_list = [f['text'] if isinstance(f, dict) else f for f in findings]
    print(f"Analyzing scan: {scan_id}")
    print(f"Findings ({len(findings_list)}): {findings_list}")
    print(f"====================================================")

    # 1. Load image and reorient to RAS
    img_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
    seg_path = os.path.join(seg_dir, f"{scan_id}.nii.gz")
    
    nib_img = nib.load(img_path)
    from nibabel.orientations import io_orientation
    img_ornt = io_orientation(nib_img.affine)
    img_r = nib_img.as_reoriented(img_ornt)
    img_data = img_r.get_fdata().transpose((2, 1, 0))[None] # shape: (1, Z, Y, X)
    
    # Load 4D segmentation mask
    nib_seg = nib.load(seg_path)
    seg_r = nib_seg.as_reoriented(img_ornt)
    seg_data = seg_r.get_fdata().transpose((0, 3, 2, 1)) # shape: (F, Z, Y, X)
    
    # Preprocess image
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    img_data = img_data.astype(np.float32)
    img_cropped, _, bbox = crop_to_nonzero(img_data, None)
    seg_cropped = seg_data[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    
    normalization = ZScoreNormalization(intensityproperties={})
    img_normalized = normalization.run(img_cropped, None)
    
    # Load pre-computed text embeddings
    cache_dir = os.path.join(os.path.dirname(dataset_json_path), "text_cache")
    text_embeddings = torch.load(os.path.join(cache_dir, f"{scan_id}.pt"), map_location=device)[None] # shape: (1, F, 2560)
    
    # Setup image tensor
    image_tensor = torch.as_tensor(img_normalized).to(device)
    seg_tensor = torch.as_tensor(seg_cropped).to(device)
    
    model_dir = os.getenv("MODEL_DIR")
    if model_dir.endswith("voxtell_v1.0"):
        model_dir = os.path.join(os.path.dirname(model_dir), "voxtell_v1.1")

    # Diagnosing Baseline
    print("\n[1] Running Baseline model inference...")
    baseline_network = load_voxtell_network(model_dir, device)
    baseline_network.eval()
    baseline_predictor = DiagnosticPredictor(model_dir, device, baseline_network)
    
    with torch.no_grad():
        baseline_logits = baseline_predictor.predict_sliding_window_return_logits(image_tensor, text_embeddings).to(device)
        baseline_probs = torch.sigmoid(baseline_logits)
        
    print("[2] Running Epoch 4 Student model inference...")
    student_network = load_voxtell_network(model_dir, device)
    checkpoint_path = "models/checkpoint_mean_teacher_latest.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    student_network.load_state_dict(checkpoint['student_state_dict'])
    student_network = student_network.to(device)
    student_network.eval()
    student_predictor = DiagnosticPredictor(model_dir, device, student_network)
    
    with torch.no_grad():
        student_logits = student_predictor.predict_sliding_window_return_logits(image_tensor, text_embeddings).to(device)
        student_probs = torch.sigmoid(student_logits)
        
    # Analyze each finding
    for f_idx, finding_name in enumerate(findings_list):
        print(f"\n----------------------------------------------------")
        print(f"Finding {f_idx}: '{finding_name}'")
        
        # Get ground truth voxel mask
        gt_mask = seg_tensor[f_idx] > 0
        gt_count = gt_mask.sum().item()
        print(f"True positive voxel count: {gt_count}")
        
        # Baseline Probabilities
        b_probs = baseline_probs[f_idx]
        b_max_all = b_probs.max().item()
        b_mean_all = b_probs.mean().item()
        
        if gt_count > 0:
            b_max_gt = b_probs[gt_mask].max().item()
            b_mean_gt = b_probs[gt_mask].mean().item()
            b_max_bg = b_probs[~gt_mask].max().item()
        else:
            b_max_gt, b_mean_gt, b_max_bg = 0.0, 0.0, b_max_all
            
        print(f"  VoxTell Baseline Probs:")
        print(f"    - Max prob (entire scan)  : {b_max_all:.6f}")
        print(f"    - Mean prob (entire scan) : {b_mean_all:.6f}")
        print(f"    - Max prob (inside GT ROI): {b_max_gt:.6f}")
        print(f"    - Mean prob (inside GT ROI): {b_mean_gt:.6f}")
        print(f"    - Max prob (outside GT)   : {b_max_bg:.6f}")
        
        # Student Probabilities
        s_probs = student_probs[f_idx]
        s_max_all = s_probs.max().item()
        s_mean_all = s_probs.mean().item()
        
        if gt_count > 0:
            s_max_gt = s_probs[gt_mask].max().item()
            s_mean_gt = s_probs[gt_mask].mean().item()
            s_max_bg = s_probs[~gt_mask].max().item()
        else:
            s_max_gt, s_mean_gt, s_max_bg = 0.0, 0.0, s_max_all
            
        print(f"  Epoch 4 Student Probs:")
        print(f"    - Max prob (entire scan)  : {s_max_all:.6f}")
        print(f"    - Mean prob (entire scan) : {s_mean_all:.6f}")
        print(f"    - Max prob (inside GT ROI): {s_max_gt:.6f}")
        print(f"    - Mean prob (inside GT ROI): {s_mean_gt:.6f}")
        print(f"    - Max prob (outside GT)   : {s_max_bg:.6f}")
        
    print(f"====================================================")

if __name__ == "__main__":
    main()
