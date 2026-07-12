import json
import re
from datasets import load_dataset

# We want roughly 15k examples to match the size of the previous python dataset
TARGET_COUNT = 15000
OUTPUT_FILE = 'oo_c_instruct.jsonl'

# Simple heuristics to detect C, C++, and Rust
def is_c_family_or_rust(instruction, output):
    text = (instruction + "\n" + output).lower()
    
    # Keyword triggers in instruction
    if re.search(r'\b(c\+\+|cpp|c language|rust)\b', instruction.lower()):
        return True
        
    # Code syntax triggers in output
    if "#include" in output and ("<stdio.h>" in output or "<iostream>" in output or "<stdint.h>" in output):
        return True
    if "fn main(" in output and "println!" in output:
        return True
    if "std::" in output or "vec!" in output:
        return True
        
    # Exclude obvious Python/Java/JS if it's borderline
    if "def " in output and ":" in output:
        return False
    if "public static void main" in output:
        return False
    if "console.log(" in output:
        return False
        
    return False

def build_dataset():
    print(f"Streaming 'sahil2801/code_instructions_120k' from HF to find {TARGET_COUNT} C/C++/Rust examples...")
    
    # We use a large, mixed code instruction dataset and stream it
    ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
    
    count = 0
    with open(OUTPUT_FILE, 'w') as f:
        for ex in ds:
            instruction = ex.get('instruction', '') or ex.get('text', '')
            output = ex.get('output', '')
            
            if not instruction or not output:
                continue
                
            if is_c_family_or_rust(instruction, output):
                # Format into our expected Alpaca text string
                text = f"### Instruction:\n{instruction}\n### Response:\n{output}<|endoftext|>"
                f.write(json.dumps({'text': text}) + '\n')
                count += 1
                
                if count % 1000 == 0:
                    print(f"Collected {count} / {TARGET_COUNT}...")
                    
            if count >= TARGET_COUNT:
                break
                
    print(f"Done! Saved {count} examples to {OUTPUT_FILE}")

if __name__ == '__main__':
    build_dataset()
