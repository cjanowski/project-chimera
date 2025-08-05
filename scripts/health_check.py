#!/usr/bin/env python3
"""
Project Chimera Health Check

Comprehensive system health check for the training environment.
Verifies dependencies, data, models, and runs basic functionality tests.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --quick  # Skip slow tests
"""

import argparse
import sys
import traceback
from pathlib import Path


def check_imports():
    """Verify all required imports work."""
    print("🔍 Checking imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ❌ Transformers import failed: {e}")
        return False
    
    try:
        from src.project_chimera.baseline import BaselineConfig, GPTDecoder
        from src.project_chimera.trainer import BaselineTrainer
        from src.project_chimera.data.preprocess import build_tokenizer, build_dataloaders
        print("  ✅ Project modules")
    except ImportError as e:
        print(f"  ❌ Project import failed: {e}")
        print("  💡 Try: export PYTHONPATH=\"$PWD/src:$PYTHONPATH\"")
        return False
    
    return True


def check_device():
    """Check device availability and compatibility."""
    print("\n🖥️  Checking device...")
    
    try:
        from src.project_chimera.utils.device import get_device, device_name
        device = get_device()
        name = device_name(device)
        print(f"  ✅ Device: {name}")
        
        if device.type == "cuda":
            import torch
            print(f"  ✅ CUDA version: {torch.version.cuda}")
            print(f"  ✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory // 1e9:.1f}GB)")
        elif device.type == "mps":
            print("  ✅ Metal Performance Shaders available")
        
        return True
    except Exception as e:
        print(f"  ❌ Device check failed: {e}")
        return False


def check_data():
    """Check data availability and preprocessing."""
    print("\n📁 Checking data...")
    
    data_dir = Path("data/ag_news")
    if not data_dir.exists():
        print("  ⚠️  AG News data not found")
        print("  💡 Run: python scripts/download_ag_news.py")
        return False
    
    # Check for required subdirectories
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  ❌ Missing {split} directory")
            return False
        
        # Check for parquet files
        parquet_files = list(split_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  ❌ No parquet files in {split_dir}")
            return False
        
        print(f"  ✅ {split}: {len(parquet_files)} parquet files")
    
    return True


def check_tokenizer():
    """Test tokenizer functionality."""
    print("\n🔤 Checking tokenizer...")
    
    try:
        from src.project_chimera.data.preprocess import build_tokenizer, TokenizerConfig
        
        tok_config = TokenizerConfig(pretrained_name="gpt2", max_length=128)
        tokenizer = build_tokenizer(tok_config)
        
        print(f"  ✅ Vocab size: {tokenizer.vocab_size}")
        print(f"  ✅ Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        
        # Test encoding/decoding
        test_text = "This is a test sentence."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        if test_text.lower() in decoded.lower():
            print("  ✅ Encoding/decoding works")
        else:
            print(f"  ⚠️  Encoding/decoding mismatch: '{test_text}' -> '{decoded}'")
        
        return True
    except Exception as e:
        print(f"  ❌ Tokenizer check failed: {e}")
        traceback.print_exc()
        return False


def check_data_loading(quick=False):
    """Test data loading pipeline."""
    print("\n📊 Checking data loading...")
    
    try:
        from src.project_chimera.data.preprocess import build_tokenizer, build_dataloaders, TokenizerConfig
        
        tok_config = TokenizerConfig(pretrained_name="gpt2", max_length=128)
        tokenizer = build_tokenizer(tok_config)
        
        limit = 32 if quick else 100
        train_loader, val_loader = build_dataloaders(
            tokenizer, 
            tok_config,
            train_limit=limit,
            val_limit=limit//2,
            batch_size=4,
            num_workers=0
        )
        
        print(f"  ✅ Train loader: {len(train_loader)} batches")
        print(f"  ✅ Val loader: {len(val_loader)} batches")
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        print(f"  ✅ Batch shape: {train_batch['input_ids'].shape}")
        
        # Verify batch contents
        assert 'input_ids' in train_batch
        assert 'attention_mask' in train_batch
        assert 'labels' in train_batch
        print("  ✅ Batch contains required keys")
        
        return True
    except Exception as e:
        print(f"  ❌ Data loading check failed: {e}")
        traceback.print_exc()
        return False


def check_model():
    """Test model creation and forward pass."""
    print("\n🤖 Checking model...")
    
    try:
        import torch
        from src.project_chimera.baseline import BaselineConfig, GPTDecoder
        from src.project_chimera.utils.device import get_device
        
        device = get_device()
        config = BaselineConfig(
            vocab_size=1000,  # Small for testing
            d_model=64,
            n_layers=2,
            n_heads=2,
            ff_dim=256,
            max_seq_len=128
        )
        
        model = GPTDecoder(config).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created: {param_count:,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        labels = input_ids.clone()
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = output['loss']
        logits = output['logits']
        
        print(f"  ✅ Forward pass: loss={loss.item():.4f}")
        print(f"  ✅ Logits shape: {logits.shape}")
        
        # Verify loss is reasonable
        if loss.item() > 0 and torch.isfinite(loss):
            print("  ✅ Loss is finite and positive")
        else:
            print(f"  ❌ Suspicious loss value: {loss.item()}")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Model check failed: {e}")
        traceback.print_exc()
        return False


def check_training(quick=False):
    """Test training loop functionality."""
    print("\n🏃 Checking training loop...")
    
    if quick:
        print("  ⏭️  Skipping training check (quick mode)")
        return True
    
    try:
        import torch
        from src.project_chimera.baseline import BaselineConfig, GPTDecoder
        from src.project_chimera.trainer import BaselineTrainer, TrainConfig
        from src.project_chimera.data.preprocess import build_tokenizer, build_dataloaders, TokenizerConfig
        from src.project_chimera.utils.device import get_device
        
        device = get_device()
        
        # Small configuration for fast testing
        tok_config = TokenizerConfig(pretrained_name="gpt2", max_length=32)
        tokenizer = build_tokenizer(tok_config)
        
        train_loader, val_loader = build_dataloaders(
            tokenizer, tok_config,
            train_limit=32, val_limit=16, batch_size=4, num_workers=0
        )
        
        model_config = BaselineConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=64, n_layers=1, n_heads=2, ff_dim=128,
            max_seq_len=32
        )
        model = GPTDecoder(model_config)
        
        train_config = TrainConfig(
            lr=1e-3, max_steps=3, log_every=1, eval_every=2,
            amp=False  # Disable AMP for testing
        )
        
        trainer = BaselineTrainer(model, train_loader, val_loader, train_config, device)
        
        # Run a few training steps
        final_metrics = trainer.train()
        
        print(f"  ✅ Training completed: val_loss={final_metrics['val_loss']:.4f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Training check failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Project Chimera health check")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    args = parser.parse_args()
    
    print("🏥 Project Chimera Health Check")
    print("=" * 50)
    
    checks = [
        ("Imports", lambda: check_imports()),
        ("Device", lambda: check_device()),
        ("Data", lambda: check_data()),
        ("Tokenizer", lambda: check_tokenizer()),
        ("Data Loading", lambda: check_data_loading(args.quick)),
        ("Model", lambda: check_model()),
        ("Training", lambda: check_training(args.quick)),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except KeyboardInterrupt:
            print(f"\n⏹️  Health check interrupted during {name}")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌ Unexpected error in {name}: {e}")
    
    print("\n" + "=" * 50)
    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("✅ System is ready for training")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("❌ Some issues need to be resolved")
        sys.exit(1)


if __name__ == "__main__":
    main()