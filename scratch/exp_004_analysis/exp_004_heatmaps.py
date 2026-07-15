import os
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

DATA_JSON = 'data/dataset.json'
SEG_DIR = 'data/raw/segmentations'
OUTPUT_DIR = 'data/analysis_experiment_004'

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_JSON, 'r') as f:
    dataset = json.load(f)

CANONICAL_SIZE = (256, 256)
heatmaps = {}

def resize_2d(img_numpy, size=(256, 256)):
    t = torch.from_numpy(img_numpy.copy()).unsqueeze(0).unsqueeze(0).float()
    t_resized = F.interpolate(t, size=size, mode='nearest')
    return t_resized.squeeze().numpy()

for split, items in dataset.items():
    group = 'Train (Sparse)' if split == 'train' else 'Val+Test (Full)'
    for item in tqdm(items, desc=f"Processing {split}"):
        filename = item['name']
        filepath = os.path.join(SEG_DIR, filename)
        if not os.path.exists(filepath):
            continue
        try:
            img = nib.load(filepath)
            mask_data = img.get_fdata(dtype=np.float32) 
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
            
        for str_idx, cat in item.get('categories', {}).items():
            f_idx = int(str_idx)
            if f_idx >= mask_data.shape[0]:
                continue
            
            if cat not in heatmaps:
                heatmaps[cat] = {
                    'Train (Sparse)': {'axial': np.zeros(CANONICAL_SIZE), 'coronal': np.zeros(CANONICAL_SIZE)},
                    'Val+Test (Full)': {'axial': np.zeros(CANONICAL_SIZE), 'coronal': np.zeros(CANONICAL_SIZE)}
                }
            
            single_mask = mask_data[f_idx]
            
            axial_proj = np.max(single_mask, axis=2)
            coronal_proj = np.max(single_mask, axis=1)
            
            # Rotate 90 degrees counter-clockwise
            axial_proj = np.rot90(axial_proj, k=1)
            coronal_proj = np.rot90(coronal_proj, k=1)
            
            axial_resized = resize_2d(axial_proj, CANONICAL_SIZE)
            coronal_resized = resize_2d(coronal_proj, CANONICAL_SIZE)
            
            heatmaps[cat][group]['axial'] += axial_resized
            heatmaps[cat][group]['coronal'] += coronal_resized

print("Finished processing masks. Saving raw data...")

with open(os.path.join(OUTPUT_DIR, 'heatmaps_raw.pkl'), 'wb') as f:
    pickle.dump(heatmaps, f)

print("Generating heatmaps...")

for cat, splits in heatmaps.items():
    for plane in ['axial', 'coronal']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for i, group in enumerate(['Train (Sparse)', 'Val+Test (Full)']):
            hm = splits[group][plane]
            im = axes[i].imshow(hm, cmap='Purples')
            axes[i].set_title(f"{group}")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"Spatial Heatmap: Category {cat} ({plane.capitalize()})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{cat}_{plane}.png"), dpi=300)
        plt.close()

print("Heatmaps generated successfully.")
