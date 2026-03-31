#!/usr/bin/env python3
"""
Test script to reproduce and fix the DynamicCache.get_max_length() error.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = "models/MobileLLM-125M"

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_1_inspect_cache_object():
    """Test 1: Inspect DynamicCache to see what methods it has."""
    print_section("Test 1: Inspecting DynamicCache methods")
    
    try:
        from transformers.cache_utils import DynamicCache
        
        cache = DynamicCache()
        print(f"DynamicCache type: {type(cache)}")
        print(f"\nAvailable methods (non-property):")
        # Avoid calling properties that might fail on empty cache
        all_attrs = dir(cache)
        for attr in sorted(all_attrs):
            if not attr.startswith('_'):
                try:
                    obj = getattr(type(cache), attr)
                    if callable(obj) and not isinstance(obj, property):
                        print(f"  - {attr}()")
                except:
                    pass
        
        # Check specific methods
        print(f"\nMethod availability:")
        print(f"  Has 'get_max_length': {hasattr(cache, 'get_max_length')}")
        print(f"  Has 'get_seq_length': {hasattr(cache, 'get_seq_length')}")
        print(f"  Has 'get_max_cache_shape': {hasattr(cache, 'get_max_cache_shape')}")
        
        return cache
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_2_try_generation():
    """Test 2: Try text generation to reproduce the error."""
    print_section("Test 2: Attempting text generation (should fail)")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        print("Tokenizing input...")
        inputs = tokenizer("Once upon a time", return_tensors="pt")
        
        # Remove token_type_ids if present (not supported by this model)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        print("Generating text...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            use_cache=True
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("✅ Generation succeeded!")
        return True
        
    except AttributeError as e:
        if "get_max_length" in str(e):
            print(f"❌ Got the expected error: {e}")
            return False
        else:
            print(f"❌ Got a different AttributeError: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"❌ Got unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_modeling_file():
    """Inspect the problematic section in modeling file."""
    print_section("Inspecting modeling_mobilellm.py")
    
    modeling_path = Path(MODEL_PATH) / "modeling_mobilellm.py"
    
    with open(modeling_path) as f:
        lines = f.readlines()
    
    print("Lines around 1289 (get_max_length issue):")
    for i in range(1285, 1295):
        if i < len(lines):
            print(f"Line {i+1}: {lines[i].rstrip()}")


def apply_fix():
    """Apply fix to the modeling file."""
    print_section("Applying Fix to modeling_mobilellm.py")
    
    modeling_path = Path(MODEL_PATH) / "modeling_mobilellm.py"
    
    with open(modeling_path) as f:
        content = f.read()
    
    # The issue: DynamicCache doesn't have get_max_length() method
    # Solution: Check if the method exists, or use appropriate alternative
    
    old_code = """            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)"""
    
    # Check if get_max_length exists; if not, don't set max_cache_length
    new_code = """            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            # Fixed: Handle DynamicCache which doesn't have get_max_length() method
            if hasattr(past_key_values, 'get_max_length') and past_key_values.get_max_length() is not None:
                max_cache_length = torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
            else:
                max_cache_length = None
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)"""
    
    if old_code in content:
        print("Found the problematic code section.")
        print("Applying fix...")
        
        # Create backup if not already exists
        backup_path = modeling_path.with_suffix('.py.bak2')
        if not backup_path.exists():
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
    print("  MobileLLM DynamicCache.get_max_length() Fix")
    print("*" * 80)
    
    # Change to recipe directory
    recipe_dir = Path(__file__).parent
    os.chdir(recipe_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Test 1: Inspect DynamicCache
    cache = test_1_inspect_cache_object()
    
    # Test 2: Try generation (should fail with get_max_length error)
    generation_works = test_2_try_generation()
    
    if not generation_works:
        # Inspect the file
        inspect_modeling_file()
        
        print_section("DIAGNOSIS")
        print("The model fails when trying to generate text due to:")
        print("  AttributeError: 'DynamicCache' object has no attribute 'get_max_length'")
        print("\nRoot cause:")
        print("  - DynamicCache in newer transformers versions doesn't have get_max_length()")
        print("  - The modeling file tries to call this non-existent method")
        print("  - Need to check if method exists before calling it")
        
        print("\n" + "-" * 80)
        response = input("\nDo you want to apply the fix to modeling_mobilellm.py? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            if apply_fix():
                print("\nFix applied! Retrying generation...")
                
                # Clear cached modules
                import sys
                modules_to_clear = [m for m in sys.modules.keys() if 'MobileLLM' in m or 'mobilellm' in m]
                for m in modules_to_clear:
                    del sys.modules[m]
                
                generation_works = test_2_try_generation()
                
                if generation_works:
                    print_section("SUCCESS!")
                    print("The model can now generate text successfully!")
    else:
        print_section("SUCCESS!")
        print("Generation works! No fix needed.")
    
    print_section("SUMMARY")
    print(f"DynamicCache inspection: {'✓' if cache else '✗'}")
    print(f"Text generation works: {'✓' if generation_works else '✗'}")


if __name__ == "__main__":
    main()
