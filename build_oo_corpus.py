import os
import json

REPO_DIR = '/tmp/oo_repo'
OUTPUT_FILE = 'oo_corpus.jsonl'
EXTENSIONS = {'.c', '.h', '.rs', '.md', '.ps1', '.txt'}

# Max chars per chunk (roughly ~250 tokens)
CHUNK_SIZE = 1000

def chunk_text(text, filename):
    # Prefix chunks with the filename so the model learns file context
    header = f"// File: {filename}\n"
    chunks = []
    
    # Simple chunking by character length, trying to split on newlines if possible
    lines = text.split('\n')
    current_chunk = header
    
    for line in lines:
        if len(current_chunk) + len(line) > CHUNK_SIZE:
            chunks.append(current_chunk)
            current_chunk = header + line + '\n'
        else:
            current_chunk += line + '\n'
            
    if len(current_chunk) > len(header):
        chunks.append(current_chunk)
        
    return chunks

def build_corpus():
    if not os.path.exists(REPO_DIR):
        print(f"Error: {REPO_DIR} does not exist. Please clone it first.")
        return
        
    print(f"Walking {REPO_DIR} for {EXTENSIONS} files...")
    
    total_files = 0
    total_chunks = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(REPO_DIR):
            # Skip git and hidden dirs
            if '.git' in root or '__pycache__' in root:
                continue
                
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXTENSIONS:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as in_f:
                            content = in_f.read()
                            
                        if not content.strip():
                            continue
                            
                        # Extract relative path for the header
                        rel_path = os.path.relpath(filepath, REPO_DIR)
                        
                        chunks = chunk_text(content, rel_path)
                        for c in chunks:
                            # Format as Alpaca-style text without instruction
                            # to match the auto-regressive layout
                            text = f"### Response:\n{c}<|endoftext|>"
                            out_f.write(json.dumps({'text': text}) + '\n')
                            total_chunks += 1
                            
                        total_files += 1
                    except Exception as e:
                        print(f"Skipping {filepath} due to read error: {e}")
                        
    print(f"Done! Processed {total_files} files into {total_chunks} chunks.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    build_corpus()
