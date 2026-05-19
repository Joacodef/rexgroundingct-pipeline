import os
import torch
import numpy as np
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Strictly isolate GPU 0 (Node policy)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import VoxTell dependencies after setting environment variables
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

def main():
    # Inject paths from .env file
    download_dir = os.getenv("MODEL_DIR")
    data_prep_dir = os.getenv("DATA_PREP_DIR")
    output_dir = os.getenv("DATA_PRED_DIR")

    # Security validation for critical environment variables
    if not all([download_dir, data_prep_dir, output_dir]):
        raise ValueError("Error: Missing environment variables in script. Check your .env file.")

    # 2. Device Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    # We will download the paper version (v1.0)
    MODEL_NAME = "voxtell_v1.0" 
    os.makedirs(download_dir, exist_ok=True)

    # 3. Get Weights
    print(f"Validating/Downloading {MODEL_NAME} weights from Hugging Face...")
    model_path = snapshot_download(
        repo_id="mrokuss/VoxTell", 
        allow_patterns=[f"{MODEL_NAME}/*", "*.json"], 
        local_dir=download_dir
    )
    voxtell_weights_dir = os.path.join(download_dir, MODEL_NAME)

    # 4. Initialize Predictor 
    # (The Qwen3-Embedding-4B text encoder will be downloaded/loaded automatically here)
    print("Initializing VoxTellPredictor...")
    predictor = VoxTellPredictor(
        model_dir=voxtell_weights_dir,
        device=device,
    )

    # 5. Data Reading and Reorientation (Critical for VoxTell)
    # Dynamic path construction using environment variable
    image_path = os.path.join(data_prep_dir, "train_6_a_2_ct.nii.gz") 
    print(f"Loading and reorienting volume to RAS: {image_path}")
    reader = NibabelIOWithReorient()
    img, img_properties = reader.read_images([image_path])

    # 6. Define Prompts (List F)
    text_prompts = ["liver", "right kidney", "left kidney", "spleen", "pancreas"]
    print(f"Prompts to evaluate: {text_prompts}")

    # 7. Inference
    print("Running Zero-Shot inference...")
    # Output: (num_prompts, x, y, z) -> Equivalent to (F, H, W, D)
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    
    print(f"Success. Output tensor shape (F, H, W, D): {voxtell_seg.shape}")

    # 8. Save final tensor
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pred_0001.npy")
    
    np.save(output_path, voxtell_seg.astype(np.uint8))
    print(f"Saved tensor ready for evaluation at: {output_path}")

if __name__ == "__main__":
    main()
