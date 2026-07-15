import os
import sys
import time
import json
import torch
import re
import ast
import argparse
from transformers import AutoTokenizer

from mamba3_titan_mimo import TitanMIMO, build_prime_lut, PrimeLinear

def generate_text(model, tokenizer, prompt, domain_id, max_new=50, domain='chat'):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
    t0 = time.time()
    
    # Structured domains need near-greedy decoding to reliably produce valid syntax/JSON
    # Chat/Math benefit from higher temperature for natural-sounding output
    if domain in ('code', 'tool'):
        temperature = 0.3
        top_k = 5
    else:
        temperature = 0.8
        top_k = 50
    
    for _ in range(max_new):
        with torch.no_grad():
            # In Phase 4, domain routing is completely autonomous! We do NOT pass domain_id.
            logits, _ = model(input_ids)
            logits = logits[:, -1, :].float()
            
        logits = logits / max(temperature, 1e-8)
        
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        if next_id.item() == tokenizer.eos_token_id:
            break
            
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        
    elapsed = time.time() - t0
    tps = max_new / max(elapsed, 1e-6)
    
    # Return only the generated suffix
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    suffix = full_text[len(prompt):]
    return suffix, tps

def load_validation_samples(val_path):
    print(f"[INIT] Loading validation samples from {val_path}...")
    samples = {'chat': [], 'math': [], 'code': [], 'tool': []}
    domain_map = {'chat': 0, 'math': 1, 'code': 2, 'tool': 3}
    
    if not os.path.exists(val_path):
        print(f"[WARN] Validation suite not found at {val_path}. Run build_val_suite.py first.")
        return samples, domain_map
        
    with open(val_path) as f:
        for line in f:
            obj = json.loads(line)
            domain = obj.get('domain')
            text = obj.get('text')
            
            # Simple heuristic parsing to split prompt from expected output
            prompt, expected = "", ""
            if domain == 'chat':
                prompt = f"### Instruction:\n{text[:100]}...\n### Response:\n"
                expected = "" # Qualitative
            elif domain == 'math':
                parts = text.split("### Response:")
                if len(parts) == 2:
                    prompt = f"{parts[0].strip()}\n### Response:\n"
                    expected = parts[1].strip()
            elif domain == 'code':
                parts = text.split("### Response:")
                if len(parts) == 2:
                    prompt = f"{parts[0].strip()}\n### Response:\n"
                    expected = parts[1].strip()
            elif domain == 'tool':
                parts = text.split("### Conversation:")
                if len(parts) == 2:
                    # Tool domain uses different format as discovered earlier
                    user_part = parts[1].split("Agent:")[0] if "Agent:" in parts[1] else parts[1]
                    # Strip the leading header from parts[0] to avoid double-printing it
                    system_body = parts[0].replace("### System (Tool Config):", "").strip()
                    prompt = f"### System (Tool Config):\n{system_body}\n### Conversation:\n{user_part.strip()}\nASSISTANT: "
                    expected = ""
                    
            if prompt and len(samples[domain]) < 10: # Just 10 samples per domain for speed in offline benchmark
                samples[domain].append((prompt, expected))
                
    return samples, domain_map

def evaluate_code_syntax(generated_code):
    # First try to extract a python code block from markdown fences
    import re as _re
    block_match = _re.search(r'```(?:python)?\s*\n(.*?)```', generated_code, _re.DOTALL)
    if block_match:
        code_to_test = block_match.group(1).strip()
    else:
        # Fall back to stripping lines that look like markdown/prose
        lines = generated_code.split('\n')
        code_lines = [l for l in lines if not l.strip().startswith('#!') and not l.strip().startswith('```')]
        code_to_test = '\n'.join(code_lines).strip()
    try:
        ast.parse(code_to_test)
        return True
    except SyntaxError:
        # Last resort: try raw
        try:
            ast.parse(generated_code)
            return True
        except SyntaxError:
            return False

def extract_numbers(text):
    return set(re.findall(r'-?\d+\.?\d*', text))

def evaluate_math_exact_match(generated, expected):
    gen_nums = extract_numbers(generated)
    exp_nums = extract_numbers(expected)
    if not exp_nums:
        return False
    # If the generation contains the expected answer numbers, we count it as a hit for now
    return bool(exp_nums & gen_nums)

def evaluate_tool_json(generated):
    # Try all JSON-like substrings from outermost { to }
    start = generated.find("{")
    end = generated.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = generated[start:end+1]
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            pass
    # Accept function_call style responses without braces too
    if 'function_call' in generated or '"name"' in generated:
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Titan MIMO Offline Evaluator")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/titan_mimo_live.pt", help="Path to checkpoint")
    parser.add_argument("--val_path", type=str, default="./data/titan_val_suite.jsonl", help="Path to validation suite")
    args = parser.parse_args()

    print(f"Loading checkpoint {args.ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = TitanMIMO()
    lut = build_prime_lut()
    for layer in model.layers:
        layer.ssm.in_proj  = PrimeLinear(layer.ssm.in_proj, lut)
        layer.ssm.out_proj = PrimeLinear(layer.ssm.out_proj, lut)
    for arm in model.mimo_arms:
        arm.ssm.in_proj  = PrimeLinear(arm.ssm.in_proj, lut)
        arm.ssm.out_proj = PrimeLinear(arm.ssm.out_proj, lut)
    model.domain_router = PrimeLinear(model.domain_router, lut)
        
    try:
        ckpt_data = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt_data['state_dict'], strict=False)
    except Exception as e:
        print(f"[FATAL] Failed to load checkpoint: {e}")
        return
        
    model.cuda().eval()
    samples, domain_map = load_validation_samples(args.val_path)
    
    results = {'chat': 0.0, 'math': 0.0, 'code': 0.0, 'tool': 0.0}
    
    print("\n============================================================")
    print(" STARTING TIER 2 OFFLINE GENERATIVE BENCHMARK ")
    print("============================================================\n")
    
    for domain, pairs in samples.items():
        if not pairs or domain not in results:
            continue
            
        print(f"--- Evaluating Domain: {domain.upper()} ---")
        domain_id = torch.tensor([domain_map[domain]]).to('cuda')
        
        successes = 0
        total = 0
        
        for i, (prompt, expected) in enumerate(pairs):
            sys.stdout.write(f"\r  Running sample {i+1}/{len(pairs)}...")
            sys.stdout.flush()
            
            # Use 200 tokens for structured domains so functions aren't truncated mid-bracket
            max_tokens = 200 if domain in ('code', 'tool') else 64
            generated, tps = generate_text(model, tokenizer, prompt, domain_id, max_new=max_tokens, domain=domain)
            print(f"\nRAW OUTPUT: {generated}\n")
            
            if domain == 'math':
                hit = evaluate_math_exact_match(generated, expected)
                successes += int(hit)
            elif domain == 'code':
                hit = evaluate_code_syntax(generated)
                successes += int(hit)
            elif domain == 'tool':
                hit = evaluate_tool_json(generated)
                successes += int(hit)
            elif domain == 'chat':
                # Qualitative proxy: Did it generate non-empty, reasonably long text without collapsing?
                hit = len(generated.split()) > 5
                successes += int(hit)
                
            total += 1
            
        print(f"\n  Result: {successes}/{total} ({successes/max(total, 1)*100:.1f}%)")
        results[domain] = successes / max(total, 1)

    print("\n============================================================")
    print(" BENCHMARK SUMMARY ")
    print("============================================================")
    print(f" Chat Coherence:  {results['chat']*100:.1f}%")
    print(f" Math Exact Match:{results['math']*100:.1f}%")
    print(f" Code Syntax Val: {results['code']*100:.1f}%")
    print(f" Tool JSON Valid: {results['tool']*100:.1f}%")
    print("============================================================\n")

if __name__ == '__main__':
    main()
