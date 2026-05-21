import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, SaveImage, EnsureChannelFirstd
from monai.data import Dataset, DataLoader, decollate_batch

# Load environment variables from .env file
load_dotenv()

# Dynamic paths from .env
DATASET_JSON = os.getenv("DATASET_JSON")
IMG_DIR = os.getenv("IMG_RAW_DIR") 
SEG_DIR = os.getenv("SEG_RAW_DIR")
TMP_PREP_DIR = os.getenv("TMP_PREP_DIR")
DATA_PREP_DIR = os.getenv("DATA_PREP_DIR")

# Determine output directory based on environment (Jumbito vs ih-condor)
if TMP_PREP_DIR:
    OUT_DIR = TMP_PREP_DIR
    print(f"[INFO] Jumbito mode detected. Writing tensors to volatile space: {OUT_DIR}")
elif DATA_PREP_DIR:
    OUT_DIR = DATA_PREP_DIR
    print(f"[INFO] ih-condor mode detected. Writing tensors to persistent storage: {OUT_DIR}")
else:
    raise ValueError("Configuration error: Neither TMP_PREP_DIR nor DATA_PREP_DIR detected in local .env.")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # 1. Define parser for CLI arguments
    parser = argparse.ArgumentParser(description="ReXGroundingCT Preprocessing Script")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], 
                        help="Dataset split to preprocess (train, val, test)")
    args = parser.parse_args()

    # 2. Read dataset.json
    print(f"Reading metadata from: {DATASET_JSON}")
    with open(DATASET_JSON, 'r') as f:
        metadata = json.load(f)
    
    entries = metadata.get(args.split, [])
    if not entries:
        print(f"[WARNING] No cases found for split '{args.split}'. Check dataset.json structure.")
        return
    
    # 3. Path mapping and existence validation
    data_dicts = []
    missing_cases = 0

    for entry in entries:
        img_path = os.path.join(IMG_DIR, entry["name"])
        seg_path = os.path.join(SEG_DIR, entry["name"])
        
        # Robust check: Ensure both raw files exist before adding to MONAI pipeline
        if not os.path.exists(img_path) or not os.path.exists(seg_path):
            print(f"[WARNING] Missing raw files for {entry['name']}. Skipping.")
            missing_cases += 1
            continue

        num_f = len(entry.get("findings", {})) 
        data_dicts.append({
            "image": img_path, 
            "label": seg_path,
            "num_findings": num_f
        })

    print(f"[INFO] Found {len(data_dicts)} valid cases. Skipped {missing_cases} missing cases.")
    
    if len(data_dicts) == 0:
        print("[ERROR] No valid data to process. Aborting.")
        return

    # 4. Spatial transformations pipeline (MONAI)
    preprocessing_pipeline = Compose([
        # Load NIfTI without automatic reordering
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        
        # Ensure channel first [1, H, W, D] ONLY for the CT image.
        # Mask already has F at index 0.
        EnsureChannelFirstd(keys=["image"]), 
        
        # Standardize orientation to RAS
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Resample to 1.5mm isotropic spacing
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=["bilinear", "nearest"]
        )
    ])

    dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # 5. Savers (resample=False prevents reverting Spacingd)
    save_img = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="ct", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False
    )
    
    # Save segmentation mask maintaining the channel dim (Finding axes)
    save_seg = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="seg", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False,
        channel_dim=0 # FORCE MONAI to know channel is at 0, don't move it to the end
    )

    print(f"Starting batch preprocessing of {len(data_dicts)} volumes to {OUT_DIR}...")
    
    # 6. Execution and validation
    for batch in tqdm(dataloader, desc=f"Preprocessing {args.split}", unit="scan"):
        for data in decollate_batch(batch):
            f_expected = data["num_findings"]
            f_real = data["label"].shape[0]
            
            # Hard validation of dimensionality (F, H, W, D) for VoxTell
            assert f_real == f_expected, (
                f"Dimensionality error in {data['image'].meta['filename_or_obj']}: "
                f"dataset.json declares {f_expected} findings, but loaded mask has {f_real} channels."
            )
            
            save_img(data["image"])
            save_seg(data["label"])

if __name__ == "__main__":
    main()