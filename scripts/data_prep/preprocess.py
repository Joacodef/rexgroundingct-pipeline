import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, SaveImage, EnsureChannelFirstd
from monai.data import Dataset, DataLoader, decollate_batch

# Load environment variables from .env file
load_dotenv()

# Dynamic paths
DATASET_JSON = os.getenv("DATASET_JSON")

# Note: Verify that the internal folders match 'images' and 'segmentations'
IMG_DIR = os.getenv("IMG_RAW_DIR") 
SEG_DIR = os.getenv("SEG_RAW_DIR")

TMP_PREP_DIR = os.getenv("TMP_PREP_DIR")
DATA_PREP_DIR = os.getenv("DATA_PREP_DIR")

if TMP_PREP_DIR:
    # Jumbito mode: Volatile variable exists. Used for fast I/O.
    OUT_DIR = TMP_PREP_DIR
    print(f"[INFO] Jumbito mode detected. Writing tensors to volatile space: {OUT_DIR}")
elif DATA_PREP_DIR:
    # ih-condor mode: TMP_PREP_DIR was removed from .env, writing directly to researcher's SSD.
    OUT_DIR = DATA_PREP_DIR
    print(f"[INFO] ih-condor mode detected. Writing tensors to persistent storage: {OUT_DIR}")
else:
    raise ValueError("Configuration error: Neither TMP_PREP_DIR nor DATA_PREP_DIR detected in local .env.")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # 1. Structured read of dataset.json
    with open(DATASET_JSON, 'r') as f:
        metadata = json.load(f)
    
    train_entries = metadata.get("train", [])
    
    # 2. Path mapping and num_findings extraction (Dimension F)
    data_dicts = []
    for entry in train_entries:
        # In case a volume has no findings, prevent len() from failing
        num_f = len(entry.get("findings", {})) 
        
        data_dicts.append({
            "image": os.path.join(IMG_DIR, entry["name"]), 
            "label": os.path.join(SEG_DIR, entry["name"]),
            "num_findings": num_f
        })

    # 3. Spatial transformations pipeline
    # LoadImaged with ensure_channel_first=True is critical:
    # - For CT (3D), adds channel dimension (1, H, W, D).
    # - For mask (4D), Nibabel reads (H, W, D, F) and MONAI permutes it to (F, H, W, D).
    preprocessing_pipeline = Compose([
        # Raw load without automatic reordering
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        
        # Add channel [1, H, W, D] ONLY to the CT image. 
        # The mask already has its F at index 0.
        EnsureChannelFirstd(keys=["image"]), 
        
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=["bilinear", "nearest"]
        )
    ])

    dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # 4. Save transformations (Decoupled from memory pipeline)
    # resample=False ensures no attempt to revert Spacingd
    save_img = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="ct", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False
    )
    save_seg = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="seg", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False
    )

    print(f"Starting batch preprocessing of {len(data_dicts)} volumes to {OUT_DIR}...")
    
    # 5. Execution and validation
    for batch in tqdm(dataloader, desc="Preprocessing scans", unit="scan"):
        for data in decollate_batch(batch):
            f_expected = data["num_findings"]
            f_real = data["label"].shape[0]
            
            # Hard validation of dimensionality (F, H, W, D) for VoxTell compatibility
            assert f_real == f_expected, (
                f"Dimensionality error in {data['image'].meta['filename_or_obj']}: "
                f"dataset.json declares {f_expected} findings, but loaded mask has {f_real} channels."
            )
            
            save_img(data["image"])
            save_seg(data["label"])

if __name__ == "__main__":
    main()
