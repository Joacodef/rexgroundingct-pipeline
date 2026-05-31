import os
import json
import subprocess
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    # 1. Create subset dataset
    print("Creating validation subset dataset...")
    dataset_json_path = os.getenv("DATASET_JSON", "data/dataset.json")
    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    # Take first 10 cases of val
    subset_data = {
        "train": [],
        "val": data["val"][:10],
        "test": []
    }

    subset_path = "data/dataset_subset.json"
    with open(subset_path, "w") as f:
        json.dump(subset_data, f, indent=4)

    print(f"Created subset file: {subset_path} with {len(subset_data['val'])} cases.")

    # 2. Run inference
    print("Running inference on GPU 2...")
    pred_dir = "data/predictions_latest_subset_teacher"
    os.makedirs(pred_dir, exist_ok=True)

    # Run voxtell_inference.py
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    cmd_inf = [
        "python",
        "scripts/voxtell/voxtell_inference.py",
        "--split", "val",
        "--dataset_json", subset_path,
        "--output_dir", pred_dir,
        "--checkpoint", "models/checkpoint_mean_teacher_latest.pth",
        "--use_teacher",
        "--tile_step_size", "0.5"
    ]

    subprocess.run(cmd_inf, check=True, env=env)

    # 3. Run evaluation
    print("Running evaluation...")
    eval_out = "data/eval_results_latest_subset_teacher.json"
    cmd_eval = [
        "python",
        "scripts/evaluate.py",
        "--split", "val",
        "--dataset_json", subset_path,
        "--pred_dir", pred_dir,
        "--output_json", eval_out
    ]

    subprocess.run(cmd_eval, check=True)
    print("Finished! Evaluation results saved to:", eval_out)

if __name__ == "__main__":
    main()
