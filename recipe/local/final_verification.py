#!/usr/bin/env python3
"""
Final comprehensive test for MobileLLM-125M with both fixes applied.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("Final Verification: MobileLLM-125M with All Fixes")
print("="*80)

print("\n1. Testing with transformers...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/MobileLLM-125M",
    trust_remote_code=True,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(
    "models/MobileLLM-125M",
    trust_remote_code=True
)

text = "The quick brown fox"
inputs = tokenizer(text, return_tensors="pt")
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']

print(f"Input: {text}")
outputs = model.generate(**inputs, max_new_tokens=15)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated}")
print("✅ Transformers generation works!\n")

print("2. Testing with lm-eval...")
from lm_eval.models.huggingface import HFLM

lm = HFLM(
    pretrained="models/MobileLLM-125M",
    trust_remote_code=True,
    device="cpu"
)
print("✅ lm-eval model loaded!\n")

print("="*80)
print("SUCCESS! Both fixes are working:")
print("  1. rope_scaling KeyError - FIXED")
print("  2. DynamicCache.get_max_length() AttributeError - FIXED")
print("="*80)
print("\nMobileLLM-125M is now ready to use with lm-eval!")
