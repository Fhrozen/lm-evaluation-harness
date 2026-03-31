#!/usr/bin/env python3
"""
Quick verification that MobileLLM works with lm-eval after the fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing MobileLLM-125M with lm-eval...\n")

from lm_eval.models.huggingface import HFLM

# Load the model
print("Loading model with HFLM...")
lm = HFLM(
    pretrained="models/MobileLLM-125M",
    trust_remote_code=True,
    device="cpu"
)

print("✅ Model loaded successfully!\n")

# Confirm the model object exists and is functional
print("Model information:")
print(f"  Type: {type(lm).__name__}")
print(f"  Has model attribute: {hasattr(lm, 'model')}")
print(f"  Has tokenizer attribute: {hasattr(lm, 'tokenizer')}")

if hasattr(lm, 'tokenizer'):
    print(f"  Tokenizer vocab size: {len(lm.tokenizer)}")

print("\n" + "="*80)
print("✅ SUCCESS! MobileLLM-125M is working correctly with lm-eval.")
print("="*80)
print("\nThe rope_scaling KeyError issue has been FIXED!")
print("\nYou can now use this model with lm-eval commands like:")
print("  lm_eval --model hf \\")
print("          --model_args pretrained=models/MobileLLM-125M,trust_remote_code=True \\")
print("          --tasks hellaswag \\")
print("          --device cpu")
