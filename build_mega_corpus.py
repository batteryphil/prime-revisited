import os
import json
from datasets import load_dataset

OUTPUT_FILE = "/data/titan_mega_corpus.jsonl"
print(f"Building Mega-Corpus at {OUTPUT_FILE}...")

def write_jsonl(f, text, domain):
    f.write(json.dumps({"text": text, "domain": domain}) + "\n")

with open(OUTPUT_FILE, "w") as f:
    # 1. TinyStories (Chat Domain)
    print("Downloading TinyStories...")
    ds_stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    count = 0
    for row in ds_stories:
        write_jsonl(f, row['text'], "chat")
        count += 1
        if count >= 300000:
            break
    print(f"  Added {count} TinyStories samples.")
    
    # 2. MetaMathQA (Math Domain)
    print("Downloading MetaMathQA...")
    ds_math = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    count = 0
    for row in ds_math:
        text = f"Question: {row['query']}\nAnswer: {row['response']}"
        write_jsonl(f, text, "math")
        count += 1
        if count >= 50000:
            break
    print(f"  Added {count} MetaMathQA samples.")
    
    # 3. Python Code Instructions (Code Domain)
    print("Downloading Python Code Instructions...")
    ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    count = 0
    for row in ds_code:
        text = f"Instruction: {row['instruction']}\nInput: {row['input']}\nOutput: {row['output']}"
        write_jsonl(f, text, "code")
        count += 1
        if count >= 20000:
            break
    print(f"  Added {count} Python Code samples.")
    
    # 4. Local C/C++ Instruct (Code Domain)
    c_instruct_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime/oo_c_instruct.jsonl"
    print(f"Merging {c_instruct_path}...")
    count = 0
    if os.path.exists(c_instruct_path):
        with open(c_instruct_path, "r") as local_f:
            for line in local_f:
                obj = json.loads(line)
                write_jsonl(f, obj.get('text', ''), "code")
                count += 1
    print(f"  Added {count} C/C++ Instruct samples.")
    
    # 5. Local OO Raw Dataset (Code Domain)
    oo_data_path = "/home/phil/.gemini/antigravity/scratch/analysis_project/oo_dataset.jsonl"
    print(f"Merging {oo_data_path}...")
    count = 0
    if os.path.exists(oo_data_path):
        with open(oo_data_path, "r") as local_f:
            for line in local_f:
                try:
                    obj = json.loads(line)
                    write_jsonl(f, obj.get('text', ''), "code")
                    count += 1
                except:
                    pass
    print(f"  Added {count} OO Raw samples.")
    
    # 6. Local Agent Corpus (Tool/Chat Domain)
    agent_corpus = "/home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime/titan_agent_corpus.jsonl"
    print(f"Merging {agent_corpus}...")
    count = 0
    if os.path.exists(agent_corpus):
        with open(agent_corpus, "r") as local_f:
            for line in local_f:
                try:
                    obj = json.loads(line)
                    domain = obj.get('domain', 'tool')
                    write_jsonl(f, obj.get('text', ''), domain)
                    count += 1
                except:
                    pass
    print(f"  Added {count} Agent Corpus samples.")

print("\nMega-Corpus successfully compiled!")
