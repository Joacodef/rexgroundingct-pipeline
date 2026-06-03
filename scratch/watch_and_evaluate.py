import os
import time
import subprocess

def main():
    target = "models/checkpoint_mean_teacher_latest.pth"
    initial_mtime = os.path.getmtime(target) if os.path.exists(target) else 0
    print(f"[{time.ctime()}] Watching {target} for updates. Initial mtime: {initial_mtime}")
    
    while True:
        time.sleep(30)
        if os.path.exists(target):
            current_mtime = os.path.getmtime(target)
            if current_mtime > initial_mtime:
                # Wait for the file to be completely written (size stability check)
                print(f"[{time.ctime()}] Checkpoint update detected! Waiting for file write to complete...")
                last_size = -1
                while True:
                    time.sleep(10)
                    curr_size = os.path.getsize(target)
                    if curr_size == last_size:
                        break
                    last_size = curr_size
                
                print(f"[{time.ctime()}] Checkpoint stable (size: {last_size / 1e9:.2f} GB). Running evaluation on GPU 2...")
                
                # Setup environment with isolated GPU
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "2"
                
                cmd = [
                    ".venv-voxtell/bin/python",
                    "scratch/run_mean_teacher_val_eval.py",
                    "--checkpoint", target,
                    "--suffix", "epoch_8"
                ]
                
                subprocess.run(cmd, env=env, check=True)
                print(f"[{time.ctime()}] Evaluation finished successfully.")
                break

if __name__ == "__main__":
    main()
