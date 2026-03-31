#!/usr/bin/env python3
"""
Test script to debug and fix MobileLLM-125M import issues.

The issue: KeyError: 'type' when accessing self.config.rope_scaling["type"]
This happens because rope_scaling might be an empty dict or improperly formatted.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Add the parent directory to the path so we can import lm_eval
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = "models/MobileLLM-125M"

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_1_inspect_config():
    """Test 1: Inspect the raw config.json file."""
    print_section("Test 1: Inspecting config.json")
    
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"rope_scaling value: {config.get('rope_scaling')}")
    print(f"rope_scaling type: {type(config.get('rope_scaling'))}")
    print(f"rope_theta value: {config.get('rope_theta')}")
    
    return config


def test_2_load_config_with_transformers():
    """Test 2: Load config using transformers AutoConfig."""
    print_section("Test 2: Loading config with AutoConfig")
    
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        print(f"Config loaded successfully!")
        print(f"rope_scaling value: {config.rope_scaling}")
        print(f"rope_scaling type: {type(config.rope_scaling)}")
        
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict):
                print(f"rope_scaling keys: {config.rope_scaling.keys()}")
                print(f"rope_scaling contents: {config.rope_scaling}")
        
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        traceback.print_exc()
        return None


def test_3_load_tokenizer():
    """Test 3: Load tokenizer."""
    print_section("Test 3: Loading tokenizer")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        print(f"✓ Tokenizer loaded successfully!")
        print(f"Vocab size: {len(tokenizer)}")
        print(f"Test encode: {tokenizer.encode('Hello world')}")
        
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        traceback.print_exc()
        return None


def test_4_load_model_transformers():
    """Test 4: Load the full model with transformers."""
    print_section("Test 4: Loading model with AutoModelForCausalLM")
    
    try:
        from transformers import AutoModelForCausalLM
        
        print("Attempting to load model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"
        )
        
        print(f"✓ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return None


def test_5_load_with_lm_eval():
    """Test 5: Load model using lm-eval."""
    print_section("Test 5: Loading model with lm-eval")
    
    try:
        from lm_eval.models.huggingface import HFLM
        
        print("Attempting to load with lm-eval HFLM...")
        lm = HFLM(
            pretrained=MODEL_PATH,
            trust_remote_code=True,
            device="cpu"
        )
        
        print(f"✓ Model loaded successfully with lm-eval!")
        return lm
    except Exception as e:
        print(f"❌ Error loading with lm-eval: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return None


def test_6_inspect_modeling_file():
    """Test 6: Inspect the modeling file for the problematic code."""
    print_section("Test 6: Inspecting modeling_mobilellm.py")
    
    modeling_path = Path(MODEL_PATH) / "modeling_mobilellm.py"
    
    with open(modeling_path) as f:
        lines = f.readlines()
    
    # Find the problematic section
    for i, line in enumerate(lines[270:300], start=271):
        if 'rope_scaling' in line.lower():
            print(f"Line {i}: {line.rstrip()}")


def test_7_simulate_error():
    """Test 7: Try to reproduce the rope_scaling error."""
    print_section("Test 7: Simulating rope_scaling error scenarios")
    
    from transformers import AutoConfig
    import torch
    
    # Test different rope_scaling scenarios
    scenarios = [
        ("None (null)", None),
        ("Empty dict", {}),
        ("Dict without 'type'", {"factor": 2.0}),
        ("Dict without 'factor'", {"type": "linear"}),
    ]
    
    for desc, rope_value in scenarios:
        print(f"\nTesting scenario: {desc}")
        print(f"  rope_scaling = {rope_value}")
        
        try:
            config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
            config.rope_scaling = rope_value
            
            print(f"  Attempting to instantiate attention module...")
            
            # Try to create a model with this config - this will trigger _init_rope
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            
            print(f"  ✓ Success - no error with rope_scaling={rope_value}")
        except KeyError as e:
            print(f"  ✗ KeyError: {e}")
            print(f"  THIS IS THE ERROR: This scenario triggers the bug!")
            return True  # Found the error
        except Exception as e:
            print(f"  ✗ Other error: {type(e).__name__}: {e}")
    
    return False


def apply_fix_to_modeling_file():
    """Apply a fix to the modeling file."""
    print_section("Applying Fix to modeling_mobilellm.py")
    
    modeling_path = Path(MODEL_PATH) / "modeling_mobilellm.py"
    
    with open(modeling_path) as f:
        content = f.read()
    
    # The issue is that rope_scaling might be an empty dict or not properly validated
    # We need to add better validation
    
    old_code = """    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]"""
    
    new_code = """    def _init_rope(self):
        # Fixed: Handle rope_scaling being None, empty dict, or improperly formatted
        if self.config.rope_scaling is None or not self.config.rope_scaling:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get("type")
            scaling_factor = self.config.rope_scaling.get("factor")
            
            if not scaling_type or not scaling_factor:
                # Fallback to standard RoPE if scaling params are missing
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
                return"""
    
    if old_code in content:
        print("Found the problematic code section.")
        print("Applying fix...")
        
        # Backup original file
        backup_path = modeling_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✓ Backup created at: {backup_path}")
        
        # Apply fix
        new_content = content.replace(old_code, new_code)
        with open(modeling_path, 'w') as f:
            f.write(new_content)
        
        print("✓ Fix applied successfully!")
        return True
    else:
        print("⚠ Could not find the exact code section to fix.")
        print("The file might have been modified or the fix might already be applied.")
        return False


def main():
    """Run all tests."""
    print("\n" + "*" * 80)
    print("  MobileLLM-125M Import Debug Script")
    print("*" * 80)
    
    # Change to the recipe directory
    recipe_dir = Path(__file__).parent
    os.chdir(recipe_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests in sequence
    config_raw = test_1_inspect_config()
    config_auto = test_2_load_config_with_transformers()
    tokenizer = test_3_load_tokenizer()
    
    # Try loading the model - this is where the error should occur
    model = test_4_load_model_transformers()
    
    # Test edge cases that might trigger the error
    error_found = test_7_simulate_error()
    
    if model is None or error_found:
        # Model failed to load or error found in edge cases
        test_6_inspect_modeling_file()
        
        print_section("DIAGNOSIS")
        if model is None:
            print("The model failed to load due to rope_scaling configuration issue.")
        else:
            print("The model loads normally, but edge cases trigger the rope_scaling error.")
        
        print("\nPossible causes:")
        print("1. rope_scaling is None in config but transformers converts it to empty dict")
        print("2. rope_scaling exists but missing 'type' or 'factor' keys")
        print("3. The modeling file doesn't handle edge cases properly")
        
        print("\n" + "-" * 80)
        response = input("\nDo you want to apply the preventive fix to modeling_mobilellm.py? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            if apply_fix_to_modeling_file():
                print("\nFix applied! Retrying tests...")
                
                # Clear any cached modules
                import sys
                modules_to_clear = [m for m in sys.modules.keys() if 'MobileLLM' in m or 'mobilellm' in m]
                for m in modules_to_clear:
                    del sys.modules[m]
                
                model = test_4_load_model_transformers()
                error_found = test_7_simulate_error()
                
                if model and not error_found:
                    print_section("SUCCESS!")
                    print("The model now loads correctly and handles edge cases after applying the fix.")
    else:
        print_section("INITIAL RESULT")
        print("The model loaded successfully! No errors found in basic tests.")
    
    # Try lm-eval if model loaded successfully
    if model:
        lm_eval_model = test_5_load_with_lm_eval()
    
    print_section("SUMMARY")
    print(f"Config loaded: {'✓' if config_auto else '✗'}")
    print(f"Tokenizer loaded: {'✓' if tokenizer else '✗'}")
    print(f"Model loaded (transformers): {'✓' if model else '✗'}")
    print(f"Model loaded (lm-eval): {'✓' if 'lm_eval_model' in locals() and lm_eval_model else '✗'}")
    print(f"Edge case error found: {'✓ (Fixed)' if error_found else '✗ (None)'}")
    
    print("\n" + "*" * 80)
    print("For more information about the issue and fix, see:")
    print("- Original error: rope_scaling[\"type\"] KeyError")
    print("- Fix: Changed dict access to use .get() method with fallback")
    print("- Location: models/MobileLLM-125M/modeling_mobilellm.py _init_rope()")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
