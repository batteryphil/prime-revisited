#!/bin/bash
source /home/phil/.gemini/antigravity/scratch/analysis_project/titan_venv/bin/activate
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
cd /home/phil/.gemini/antigravity/scratch/analysis_project/mamba-prime
python3 mamba3_prime_native.py
