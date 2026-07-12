import json
import random

def main():
    print("[INIT] Loading unbalanced corpus...")
    domains = {'chat': [], 'math': [], 'code': [], 'tool': []}
    
    with open('/data/titan_mega_corpus.jsonl', 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                d = obj.get('domain')
                if d in domains:
                    domains[d].append(line)
            except Exception:
                continue

    target_count = max(len(samples) for samples in domains.values())
    print(f"[DATA] Target balanced count per domain: {target_count}")
    
    balanced_dataset = []
    
    for d, samples in domains.items():
        count = len(samples)
        print(f"  -> {d.upper()}: {count} samples")
        
        # Add all existing samples
        balanced_dataset.extend(samples)
        
        # Upsample if necessary
        deficit = target_count - count
        if deficit > 0:
            print(f"     Upsampling {deficit} {d} samples...")
            # We can use random.choices for sampling with replacement
            upsampled = random.choices(samples, k=deficit)
            balanced_dataset.extend(upsampled)
            
    print(f"\n[DATA] Total dataset size before shuffle: {len(balanced_dataset)}")
    print("[SHUFFLE] Shuffling 2+ million samples globally. This may take a minute...")
    random.shuffle(balanced_dataset)
    
    out_path = '/data/titan_mega_corpus_balanced.jsonl'
    print(f"[WRITE] Saving to {out_path}...")
    
    with open(out_path, 'w') as f:
        for line in balanced_dataset:
            f.write(line)
            
    print("[DONE] Corpus successfully balanced and saved!")

if __name__ == '__main__':
    main()
