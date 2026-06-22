import json

with open("samples_mamba3_native.json") as f:
    data = json.load(f)

last = data[-1]
print("=" * 60)
print(f"COHERENCE TEST — Step {last['step']}")
print("=" * 60)

for i, (prompt, sample) in enumerate(zip(last["prompts"], last["samples"])):
    print(f"\nPROMPT {i+1}:\n  {prompt.strip()}")
    print(f"\nOUTPUT:\n{sample.strip()}")
    print("\n" + "-" * 60)
