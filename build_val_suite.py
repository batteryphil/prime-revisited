import os
import json
from datasets import load_dataset

OUTPUT_FILE = "/data/titan_val_suite.jsonl"
print(f"Building Validation Suite at {OUTPUT_FILE}...")

def write_jsonl(f, text, domain):
    f.write(json.dumps({"text": text, "domain": domain}) + "\n")

with open(OUTPUT_FILE, "w") as f:
    # 1. TinyStories (Chat Domain) - Validation Split
    print("Downloading TinyStories (Validation)...")
    ds_stories = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    count = 0
    for row in ds_stories:
        write_jsonl(f, row['text'], "chat")
        count += 1
        if count >= 250:
            break
    print(f"  Added {count} TinyStories validation samples.")
    
    # 2. MetaMathQA (Math Domain) - Skip first 350,000 train samples
    print("Downloading MetaMathQA...")
    ds_math = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    count = 0
    skipped = 0
    for row in ds_math:
        if skipped < 350000:
            skipped += 1
            continue
        text = f"Question: {row['query']}\nAnswer: {row['response']}"
        write_jsonl(f, text, "math")
        count += 1
        if count >= 250:
            break
    print(f"  Added {count} MetaMathQA held-out samples.")
    
    # 3. MBPP (Code Domain) - Validation Split
    print("Downloading MBPP (Validation)...")
    ds_code = load_dataset("google-research-datasets/mbpp", split="validation", streaming=True)
    count = 0
    for row in ds_code:
        text = f"Instruction: {row['text']}\nOutput: {row['code']}"
        write_jsonl(f, text, "code")
        count += 1
        if count >= 250:
            break
    print(f"  Added {count} MBPP Validation samples.")
    
    # 4. Glaive Function Calling (Tool Domain) - Skip first 110,000 samples
    print("Downloading Glaive Function Calling...")
    ds_tool = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    count = 0
    skipped = 0
    for row in ds_tool:
        if skipped < 110000:
            skipped += 1
            continue
        sys_msg = row.get('system', '')
        chat = row.get('chat', '')
        text = f"### System (Tool Config):\n{sys_msg}\n### Conversation:\n{chat}"
        write_jsonl(f, text, "tool")
        count += 1
        if count >= 250:
            break
    print(f"  Added {count} Glaive held-out samples.")

print(f"\nValidation Suite successfully compiled to {OUTPUT_FILE}!")
