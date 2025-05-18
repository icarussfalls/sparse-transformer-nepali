from datasets import load_dataset
from config import get_config
import json

def save_raw_dataset():
    config = get_config()
    ds_raw = load_dataset(
        f"{config['data_source']}", 
        f"{config['lang_src']}-{config['lang_tgt']}", 
        split='train'
    )

    # Convert to list of dicts for easy saving
    data = [item for item in ds_raw]

    # Save as JSONL (one JSON object per line)
    with open("ds_raw.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} samples to ds_raw.jsonl")

if __name__ == "__main__":
    save_raw_dataset()