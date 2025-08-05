#!/usr/bin/env python3
"""
Validation script that checks if overfitting mitigation changes were implemented correctly
by examining the source files directly (no torch import required).
"""

import re
from pathlib import Path


def check_trainer_improvements():
    """Check that trainer.py has the required overfitting mitigation features."""
    trainer_path = Path("src/project_chimera/trainer.py")
    content = trainer_path.read_text()
    
    checks = [
        ("early_stopping_patience", "Early stopping patience parameter"),
        ("early_stopping_min_delta", "Early stopping min delta parameter"),
        ("val_check_interval", "Validation check interval parameter"),
        ("lr_scheduler_type", "Learning rate scheduler type parameter"),
        ("label_smoothing", "Label smoothing parameter"),
        ("_create_scheduler", "Scheduler creation method"),
        ("_warmup_lr", "Learning rate warmup method"),
        ("_check_early_stopping", "Early stopping check method"),
        ("_label_smoothing_loss", "Label smoothing loss method"),
        ("self.early_stopped", "Early stopping state tracking"),
        ("self.best_val_loss", "Best validation loss tracking"),
        ("self.patience_counter", "Patience counter tracking"),
    ]
    
    print("Checking trainer.py improvements:")
    all_passed = True
    
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}")
            all_passed = False
    
    # Check for enhanced training loop
    if "while self._step < self.cfg.max_steps and not self.early_stopped:" in content:
        print("  ✓ Enhanced training loop with early stopping")
    else:
        print("  ✗ Enhanced training loop with early stopping")
        all_passed = False
        
    if "val_check_interval" in content and "self._check_early_stopping" in content:
        print("  ✓ Frequent validation monitoring")
    else:
        print("  ✗ Frequent validation monitoring")
        all_passed = False
    
    return all_passed


def check_script_improvements():
    """Check that train_agnews.py has improved hyperparameters."""
    script_path = Path("scripts/train_agnews.py")
    content = script_path.read_text()
    
    print("\nChecking train_agnews.py improvements:")
    all_passed = True
    
    # Check model capacity reductions
    if "n_layers: int = 2" in content:
        print("  ✓ Reduced model layers from 4 to 2")
    else:
        print("  ✗ Reduced model layers from 4 to 2")
        all_passed = False
        
    if "ff_dim: int = 512" in content:
        print("  ✓ Reduced FF dimension from 1024 to 512")
    else:
        print("  ✗ Reduced FF dimension from 1024 to 512")
        all_passed = False
    
    # Check dropout increase
    if re.search(r"dropout: float = 0\.[3-9]", content):
        print("  ✓ Increased dropout rate")
    else:
        print("  ✗ Increased dropout rate")
        all_passed = False
    
    # Check learning rate reduction
    if "lr: float = 1e-4" in content:
        print("  ✓ Reduced learning rate")
    else:
        print("  ✗ Reduced learning rate")
        all_passed = False
    
    # Check weight decay increase
    if "weight_decay: float = 0.1" in content:
        print("  ✓ Increased weight decay")
    else:
        print("  ✗ Increased weight decay")
        all_passed = False
        
    # Check max steps increase
    if "max_steps: int = 1000" in content:
        print("  ✓ Increased max steps for early stopping")
    else:
        print("  ✗ Increased max steps for early stopping")
        all_passed = False
    
    return all_passed


def check_data_augmentation():
    """Check that data augmentation features were added."""
    preprocess_path = Path("src/project_chimera/data/preprocess.py")
    content = preprocess_path.read_text()
    
    print("\nChecking data augmentation improvements:")
    all_passed = True
    
    checks = [
        ("augment_data: bool", "Data augmentation toggle"),
        ("augment_prob: float", "Augmentation probability parameter"),
        ("max_augmentations_per_text", "Max augmentations parameter"),
        ("apply_text_augmentation", "Text augmentation function"),
        ("synonym_replacement", "Synonym replacement augmentation"),
        ("word_dropout", "Word dropout augmentation"),
        ("word_shuffle", "Word shuffle augmentation"),
        ("if split == \"train\" and cfg.augment_data:", "Training-only augmentation"),
    ]
    
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ✗ {description}")
            all_passed = False
    
    return all_passed


def print_parameter_comparison():
    """Print before/after comparison of key parameters."""
    print("\n" + "="*60)
    print("PARAMETER COMPARISON (Before → After)")
    print("="*60)
    
    comparisons = [
        ("Model Layers", "4", "2", "50% reduction in depth"),
        ("FF Dimension", "1024", "512", "50% reduction in width"),
        ("Dropout Rate", "0.1", "0.4", "4x stronger regularization"),
        ("Learning Rate", "3e-4", "1e-4", "3x slower learning"),
        ("Weight Decay", "0.01", "0.1", "10x stronger L2 penalty"),
        ("Max Steps", "200", "1000", "5x more steps for early stopping"),
        ("Validation Checks", "Every 100 steps", "Every 5 steps", "20x more frequent"),
        ("Label Smoothing", "None", "0.1", "Reduces overconfident predictions"),
        ("Data Augmentation", "None", "3 techniques", "~2-3x effective dataset size"),
        ("Early Stopping", "None", "Patience=10", "Automatic overfitting prevention"),
    ]
    
    print(f"{'Parameter':<20} {'Before':<15} {'After':<15} {'Impact'}")
    print("-" * 80)
    
    for param, before, after, impact in comparisons:
        print(f"{param:<20} {before:<15} {after:<15} {impact}")
    
    print("="*60)


def main():
    """Main validation function."""
    print("Validating overfitting mitigation implementation...")
    print("="*60)
    
    results = []
    results.append(check_trainer_improvements())
    results.append(check_script_improvements()) 
    results.append(check_data_augmentation())
    
    print("\n" + "="*60)
    if all(results):
        print("✅ ALL CHECKS PASSED!")
        print("Overfitting mitigation successfully implemented.")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("Please review the implementation.")
    
    print_parameter_comparison()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run training with: python scripts/train_agnews.py --run_dense")
    print("2. Monitor the training progress:")
    print("   - Training loss should remain > 0.1")
    print("   - Validation loss should track with training loss")
    print("   - Early stopping should trigger if overfitting occurs")
    print("3. Compare results with the original overfitting run")
    print("4. Adjust hyperparameters if needed based on results")
    print("="*60)


if __name__ == "__main__":
    main()