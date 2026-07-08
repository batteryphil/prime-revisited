import torch
from transformers import AutoTokenizer
from mamba3_titan_mimo import TitanMIMO, CONFIG, build_prime_lut, PrimeLinear
import json

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Titan MIMO 650M Evaluation Script")
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/titan_mimo_live.pt', help='Path to model checkpoint')
    parser.add_argument('--corpus_path', type=str, default='./data/titan_mega_corpus.jsonl', help='Path to training mega corpus jsonl for extraction')
    args = parser.parse_args()

    print("Loading Phase 4 Autonomous Router Model...")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = TitanMIMO().to(device)
    lut = build_prime_lut()
    
    # Wrap all discrete components
    for layer in model.layers:
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
    for arm in model.mimo_arms:
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
    model.domain_router = PrimeLinear(model.domain_router, lut)
        
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint not found at {args.ckpt_path}")
        sys.exit(1)
        
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    print("Extracting raw samples from mega corpus...")
    tool_sample = None
    code_sample = None
    math_sample = None
    chat_sample = None
    
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus not found at {args.corpus_path}")
        sys.exit(1)
        
    with open(args.corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            d = obj.get('domain')
            if d == 'tool' and tool_sample is None and "<tool_call>" in obj['text']:
                tool_sample = obj['text']
            elif d == 'code' and code_sample is None:
                code_sample = obj['text']
            elif d == 'math' and math_sample is None:
                math_sample = obj['text']
            elif d == 'chat' and chat_sample is None:
                chat_sample = obj['text']
                
            if tool_sample and code_sample and math_sample and chat_sample:
                break
                
    def run_eval(name, full_text, split_token):
        print(f"\n{'='*50}\n {name.upper()} DOMAIN EVALUATION (Autonomous Routing)\n{'='*50}")
        if split_token not in full_text:
            print("Split token not found in sample.")
            return
            
        prompt = full_text.split(split_token)[0] + split_token
        print("--- PROMPT ---")
        print(prompt.strip())
        print("--- AUTONOMOUS GENERATION ---")
        
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_ids = inputs[0].tolist()
        for _ in range(120):
            with torch.no_grad():
                # NO domain_id is passed! The model must route itself!
                logits, _ = model(torch.tensor([gen_ids], device=device))
                next_token = logits[0, -1].argmax().item()
                gen_ids.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break
        print(tokenizer.decode(gen_ids[len(inputs[0]):], skip_special_tokens=True).strip())

    run_eval("Chat", chat_sample, "Response:")
    run_eval("Math", math_sample, "Answer:")
    run_eval("Code", code_sample, "Output:")
    run_eval("Tool", tool_sample, "ASSISTANT:")

if __name__ == "__main__":
    main()
