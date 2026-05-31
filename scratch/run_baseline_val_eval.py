import os
import subprocess
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    
    # 1. Run baseline inference
    print("Running baseline inference on all 50 validation cases on GPU 2...")
    pred_dir = "data/predictions_baseline_validation"
    os.makedirs(pred_dir, exist_ok=True)
    
    # Dynamic visible devices for SLURM compatibility
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    
    cmd_inf = [
        "python",
        "scripts/voxtell/voxtell_inference.py",
        "--split", "val",
        "--output_dir", pred_dir,
        "--tile_step_size", "0.5"
    ]
    
    subprocess.run(cmd_inf, check=True, env=env)
    
    # 2. Run evaluation
    print("Running evaluation...")
    eval_out = "data/eval_results_baseline_validation.json"
    cmd_eval = [
        "python",
        "scripts/evaluate.py",
        "--split", "val",
        "--pred_dir", pred_dir,
        "--output_json", eval_out
    ]
    
    subprocess.run(cmd_eval, check=True)
    print("Finished baseline validation replication! Results saved to:", eval_out)

if __name__ == "__main__":
    main()
