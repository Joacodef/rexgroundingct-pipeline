import sys
import os
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

def main():
    path = "wandb/offline-run-20260529_213843-5n8xzgso/run-5n8xzgso.wandb"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Opening datastore: {path}")
    ds = DataStore()
    ds.open_for_scan(path)

    record_count = 0
    history_count = 0
    exit_record = None
    summary_record = None
    
    while True:
        res = ds.scan_record()
        if res is None:
            break
        offset, data = res
        record_count += 1
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        
        # Check record type
        oneof_type = record.WhichOneof("record_type")
        if oneof_type == "history":
            history_count += 1
            if history_count <= 5 or history_count % 100 == 0:
                print(f"History record {history_count}: {record.history}")
        elif oneof_type == "exit":
            exit_record = record.exit
            print(f"Exit record: {exit_record}")
        elif oneof_type == "summary":
            summary_record = record.summary
            print(f"Summary record: {summary_record}")
            
    print(f"Finished. Total records: {record_count}, History records (metrics logged): {history_count}")
    if exit_record:
        print(f"Exit code: {exit_record.exit_code}")
    else:
        print("No exit record found in datastore (the process might have crashed or been killed abruptly).")

if __name__ == "__main__":
    main()
