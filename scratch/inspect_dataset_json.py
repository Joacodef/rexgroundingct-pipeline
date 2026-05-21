import os
import json
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)
    dataset_json = os.getenv("DATASET_JSON")
    
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/inspect_dataset.log"
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("=" * 70 + "\n")
        log_file.write("      DATASET.JSON STRUCTURE INSPECTION\n")
        log_file.write("=" * 70 + "\n")
        
        if not dataset_json or not os.path.exists(dataset_json):
            log_file.write(f"[ERROR] Dataset JSON not found at: {dataset_json}\n")
            return
            
        with open(dataset_json, 'r') as f:
            metadata = json.load(f)
            
        log_file.write(f"Keys in dataset.json: {list(metadata.keys())}\n\n")
        
        for split in ["train", "val", "test"]:
            entries = metadata.get(split, [])
            log_file.write(f"Split '{split}': count = {len(entries)}\n")
            if entries:
                log_file.write(f"First entry in '{split}':\n")
                log_file.write(json.dumps(entries[0], indent=2) + "\n\n")
                
    print(f"Dataset structure written successfully to {log_path}")

if __name__ == "__main__":
    main()
