#!/usr/bin/env python3
"""
Convert SentencePiece/TikToken tokenizer to HuggingFace tokenizers-compatible format.
This fixes RecursionError issues in newer transformers versions by creating a 
PreTrainedTokenizerFast-based tokenizer.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional

try:
    from transformers import LlamaTokenizer, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, normalizers
    from tokenizers.trainers import BpeTrainer
except ImportError as e:
    print(f"Error: Missing required packages. Please install:")
    print("pip install transformers tokenizers sentencepiece")
    raise e


def backup_old_tokenizer(model_path: Path) -> Path:
    """Move old tokenizer files to a backup folder."""
    backup_dir = model_path / "old_tokenizer_backup"
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    ]
    
    backed_up = []
    for file in files_to_backup:
        src = model_path / file
        if src.exists():
            dst = backup_dir / file
            if not dst.exists():  # Don't overwrite existing backups
                shutil.copy2(src, dst)
                backed_up.append(file)
                print(f"  Backed up: {file}")
    
    if backed_up:
        print(f"✓ Backed up {len(backed_up)} files to: {backup_dir}")
    
    return backup_dir


def load_sentencepiece_vocab(tokenizer_model_path: Path) -> dict:
    """Load vocabulary from SentencePiece model."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(str(tokenizer_model_path))
        
        vocab = {}
        for i in range(sp.GetPieceSize()):
            token = sp.IdToPiece(i)
            vocab[token] = i
        
        return vocab, sp
    except ImportError:
        print("Warning: sentencepiece not installed. Attempting alternative method...")
        return None, None


def create_hf_tokenizer_from_sentencepiece(
    model_path: Path,
    tokenizer_model_file: str = "tokenizer.model",
) -> Optional[PreTrainedTokenizerFast]:
    """
    Create a HuggingFace tokenizers-compatible tokenizer from SentencePiece model.
    
    Args:
        model_path: Path to the model directory
        tokenizer_model_file: Name of the tokenizer model file
    
    Returns:
        PreTrainedTokenizerFast tokenizer or None if conversion fails
    """
    tokenizer_model_path = model_path / tokenizer_model_file
    config_path = model_path / "tokenizer_config.json"
    model_config_path = model_path / "config.json"
    
    if not tokenizer_model_path.exists():
        print(f"Error: Tokenizer model not found at {tokenizer_model_path}")
        return None
    
    print(f"\n📖 Loading SentencePiece model directly from: {tokenizer_model_path}")
    
    # Load configuration to get special tokens and vocab size
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    model_config = {}
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
    
    # Get special token IDs
    bos_token_id = model_config.get('bos_token_id', 1)
    eos_token_id = model_config.get('eos_token_id', 2)
    vocab_size = model_config.get('vocab_size', 32000)
    
    print(f"  Config: vocab_size={vocab_size}, bos_id={bos_token_id}, eos_id={eos_token_id}")
    
    # Load vocabulary directly from SentencePiece without using LlamaTokenizer
    try:
        vocab, sp_model = load_sentencepiece_vocab(tokenizer_model_path)
        if vocab is None or sp_model is None:
            print("Error: Could not load SentencePiece vocabulary")
            return None
        
        print(f"✓ Loaded vocabulary with {len(vocab)} tokens")
        
        # Build vocab list sorted by token ID
        vocab_list = [''] * len(vocab)
        for token, idx in vocab.items():
            vocab_list[idx] = token
        
        # Get special tokens from vocab
        bos_token = vocab_list[bos_token_id] if bos_token_id < len(vocab_list) else "<s>"
        eos_token = vocab_list[eos_token_id] if eos_token_id < len(vocab_list) else "</s>"
        unk_token = vocab_list[0] if len(vocab_list) > 0 else "<unk>"
        pad_token = unk_token
        
        print(f"  Special tokens: BOS={bos_token} (id={bos_token_id}), EOS={eos_token} (id={eos_token_id})")
        print(f"                  UNK={unk_token}, PAD={pad_token}")
        
        # Create tokenizer using the tokenizers library directly
        print("\n🔄 Building tokenizer from SentencePiece model...")
        
        # Import required components from tokenizers
        from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
        from tokenizers.models import Unigram
        
        # Build the Unigram model with vocabulary
        # SentencePiece uses Unigram language model
        vocab_scores = []
        for i in range(sp_model.GetPieceSize()):
            piece = sp_model.IdToPiece(i)
            score = sp_model.GetScore(i)
            vocab_scores.append((piece, score))
        
        unigram_model = Unigram(vocab_scores, unk_id=0)
        tokenizer_obj = Tokenizer(unigram_model)
        
        # Add normalizer (SentencePiece typically doesn't normalize)
        tokenizer_obj.normalizer = normalizers.Sequence([])
        
        # Add pre-tokenizer (SentencePiece uses Metaspace)
        # Note: API varies by tokenizers version, try different parameter combinations
        try:
            tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        except TypeError:
            # Newer versions might use different API
            try:
                tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
            except TypeError:
                # Fallback to basic Metaspace
                tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
        
        # Add decoder
        try:
            tokenizer_obj.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
        except TypeError:
            # Newer versions might use different API
            try:
                tokenizer_obj.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
            except TypeError:
                # Fallback to basic Metaspace
                tokenizer_obj.decoder = decoders.Metaspace(replacement="▁")
        
        print("✓ Successfully built tokenizer object")
        
        # Create PreTrainedTokenizerFast
        print("\n🔄 Creating PreTrainedTokenizerFast...")
        
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            model_max_length=2048,
            clean_up_tokenization_spaces=False,
        )
        
        print("✓ Successfully created PreTrainedTokenizerFast")
        
        return fast_tokenizer
        
    except Exception as e:
        print(f"Error building tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_tokenizer(model_path: str, backup: bool = True) -> bool:
    """
    Main conversion function.
    
    Args:
        model_path: Path to the model directory (relative or absolute)
        backup: Whether to backup old tokenizer files
    
    Returns:
        True if successful, False otherwise
    """
    # Get the directory of this script
    script_dir = Path(__file__).parent
    model_path = Path(model_path)
    
    # If relative path, make it relative to script directory
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return False
    
    print(f"=" * 70)
    print(f"Converting tokenizer for: {model_path.name}")
    print(f"=" * 70)
    
    # Backup old tokenizer files
    if backup:
        backup_old_tokenizer(model_path)
    
    # Create new fast tokenizer
    fast_tokenizer = create_hf_tokenizer_from_sentencepiece(model_path)
    
    if fast_tokenizer is None:
        print("\n❌ Failed to create fast tokenizer")
        return False
    
    # Save the new tokenizer
    print(f"\n💾 Saving new tokenizer to: {model_path}")
    try:
        fast_tokenizer.save_pretrained(str(model_path))
        print("✓ Successfully saved new tokenizer files")
        
        # Update tokenizer_config.json to use the fast tokenizer class
        config_path = model_path / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update to use the fast tokenizer class
            config['tokenizer_class'] = 'PreTrainedTokenizerFast'
            
            # Fix model_max_length if it's unreasonably large
            if 'model_max_length' in config and config['model_max_length'] > 1e10:
                config['model_max_length'] = 2048
                print("  ⚠ Fixed unreasonably large model_max_length to 2048")
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✓ Updated tokenizer_config.json")
        
        print("\n" + "=" * 70)
        print("✅ Conversion completed successfully!")
        print("=" * 70)
        print("\nYou can now use this tokenizer with:")
        print(f"  from transformers import AutoTokenizer")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{model_path}')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error saving tokenizer: {e}")
        return False


def test_tokenizer(model_path: str):
    """Test the converted tokenizer."""
    from transformers import AutoTokenizer
    
    print(f"\n🧪 Testing tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test encoding/decoding
        test_text = "Hello, world! This is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"✓ Tokenizer loaded successfully: {type(tokenizer).__name__}")
        print(f"  Test text: {test_text}")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Decoded: {decoded}")
        print(f"  Vocab size: {len(tokenizer)}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert SentencePiece/TikToken tokenizer to HuggingFace format"
    )
    parser.add_argument(
        'model_path',
        nargs='?',
        default='models/MobileLLM-125M',
        help='Path to model directory (default: models/MobileLLM-125M)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not backup old tokenizer files'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the tokenizer after conversion'
    )
    
    args = parser.parse_args()
    
    # Convert the tokenizer
    success = convert_tokenizer(args.model_path, backup=not args.no_backup)
    
    # Test if requested
    if success and args.test:
        script_dir = Path(__file__).parent
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = script_dir / model_path
        test_tokenizer(str(model_path))
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
