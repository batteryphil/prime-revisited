import json
import os
from datasets import load_dataset
from huggingface_hub import login

_tok_path = os.path.expanduser('~/.hf_token')
if os.path.exists(_tok_path):
    login(token=open(_tok_path).read().strip())
elif 'HF_TOKEN' in os.environ:
    login(token=os.environ['HF_TOKEN'])

print("[DATA] Building 2B Titan Agent Corpus...")
output_file = 'titan_agent_corpus.jsonl'

# Track total samples
total_written = 0
limit_per_domain = 250000  # Massive dataset for the 14-day training block

with open(output_file, 'w') as out:
    # ── 1. Math Data ────────────────────────────────────────────────────────
    print("[DATA] Loading MathInstruct...")
    try:
        math_ds = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
        count = 0
        for row in math_ds:
            if count >= limit_per_domain: break
            # math_instruct uses 'instruction' and 'output'
            if 'instruction' in row and 'output' in row:
                text = f"### Instruction:\n{row['instruction']}\n### Response:\n{row['output']}"
                out.write(json.dumps({'text': text, 'domain': 'math'}) + '\n')
                count += 1
        total_written += count
        print(f"[DATA] Wrote {count} Math samples.")
    except Exception as e:
        print(f"[WARN] Failed to load Math: {e}")

    # ── 2. Code Data ────────────────────────────────────────────────────────
    print("[DATA] Loading Code Instructions...")
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        count = 0
        for row in code_ds:
            if count >= limit_per_domain: break
            prompt = row.get('instruction', '')
            text = f"### Instruction:\n{prompt}\n### Response:\n{row['output']}"
            out.write(json.dumps({'text': text, 'domain': 'code'}) + '\n')
            count += 1
        total_written += count
        print(f"[DATA] Wrote {count} Code samples.")
    except Exception as e:
        print(f"[WARN] Failed to load Code: {e}")

    # ── 3. Tool Use Data ────────────────────────────────────────────────────
    print("[DATA] Loading Glaive Tool Calling...")
    try:
        tool_ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
        count = 0
        for row in tool_ds:
            if count >= limit_per_domain: break
            sys_msg = row.get('system', '')
            chat = row.get('chat', '')
            text = f"### System (Tool Config):\n{sys_msg}\n### Conversation:\n{chat}"
            out.write(json.dumps({'text': text, 'domain': 'tool'}) + '\n')
            count += 1
        total_written += count
        print(f"[DATA] Wrote {count} Tool Use samples.")
    except Exception as e:
        print(f"[WARN] Failed to load Tool Use: {e}")

    # ── 4. Chat / General Instruction ───────────────────────────────────────
    print("[DATA] Loading UltraChat...")
    try:
        chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        count = 0
        for row in chat_ds:
            if count >= limit_per_domain: break
            messages = row.get('messages', [])
            convo = ""
            for m in messages:
                role = "Instruction" if m['role'] == 'user' else "Response"
                convo += f"### {role}:\n{m['content']}\n"
            out.write(json.dumps({'text': convo.strip(), 'domain': 'chat'}) + '\n')
            count += 1
        total_written += count
        print(f"[DATA] Wrote {count} Chat samples.")
    except Exception as e:
        print(f"[WARN] Failed to load Chat: {e}")

print(f"\n[DONE] Wrote {total_written} multi-domain agent samples to {output_file}.")
