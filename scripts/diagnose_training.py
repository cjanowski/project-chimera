#!/usr/bin/env python3
"""
Training Diagnostics Script

Automated analysis of training results to detect common issues like overfitting,
poor convergence, or configuration problems.

Usage:
    python scripts/diagnose_training.py runs/experiment.json
    python scripts/diagnose_training.py $(ls -t runs/*.json | head -1)  # Latest run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


class TrainingDiagnostics:
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.data = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load and validate results file."""
        try:
            with open(self.results_file) as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in results file: {e}")
            sys.exit(1)
    
    def run_all_diagnostics(self) -> None:
        """Run complete diagnostic suite."""
        print(f"🔍 Diagnosing training results: {self.results_file.name}")
        print("=" * 60)
        
        # Basic info
        self._print_experiment_info()
        print()
        
        # Core diagnostics
        issues_found = []
        issues_found.extend(self._check_overfitting())
        issues_found.extend(self._check_convergence())
        issues_found.extend(self._check_configuration())
        issues_found.extend(self._check_model_capacity())
        
        # Summary
        print("\n" + "=" * 60)
        if issues_found:
            print(f"⚠️  Found {len(issues_found)} potential issues:")
            for i, issue in enumerate(issues_found, 1):
                print(f"  {i}. {issue}")
            print("\n💡 See TROUBLESHOOTING.md for solutions")
        else:
            print("✅ No obvious issues detected - training looks healthy!")
    
    def _print_experiment_info(self) -> None:
        """Print basic experiment information."""
        config = self.data.get('config_common', {})
        results = self.data.get('results', {})
        
        print("📊 Experiment Overview:")
        print(f"  • Run tag: {config.get('run_tag', 'N/A')}")
        print(f"  • Device: {results.get('dense', {}).get('device', 'N/A')}")
        print(f"  • Model: {config.get('d_model', 'N/A')}d x {config.get('n_layers', 'N/A')}L")
        print(f"  • Training data: {config.get('train_limit', 'N/A')} samples")
        print(f"  • Validation data: {config.get('val_limit', 'N/A')} samples")
        print(f"  • Training steps: {config.get('max_steps', 'N/A')}")
    
    def _check_overfitting(self) -> List[str]:
        """Check for overfitting indicators."""
        issues = []
        
        for variant in ['dense', 'moe']:
            if variant not in self.data.get('results', {}):
                continue
                
            metrics = self.data['results'][variant].get('final_metrics', {})
            val_loss = metrics.get('val_loss')
            
            if val_loss is None:
                issues.append(f"{variant.upper()}: No validation loss recorded")
                continue
            
            # Check for suspiciously low validation loss
            if val_loss < 0.001:
                issues.append(f"{variant.upper()}: Validation loss extremely low ({val_loss:.6f}) - severe overfitting likely")
            elif val_loss < 0.01:
                issues.append(f"{variant.upper()}: Validation loss very low ({val_loss:.4f}) - possible overfitting")
            elif val_loss < 0.1:
                print(f"ℹ️  {variant.upper()}: Low validation loss ({val_loss:.4f}) - monitor for overfitting")
            else:
                print(f"✅ {variant.upper()}: Validation loss looks healthy ({val_loss:.4f})")
        
        return issues
    
    def _check_convergence(self) -> List[str]:
        """Check for convergence issues."""
        issues = []
        
        for variant in ['dense', 'moe']:
            if variant not in self.data.get('results', {}):
                continue
                
            metrics = self.data['results'][variant].get('final_metrics', {})
            val_loss = metrics.get('val_loss')
            
            if val_loss is None:
                continue
            
            # Check for poor convergence (very high loss)
            if val_loss > 10.0:
                issues.append(f"{variant.upper()}: Very high validation loss ({val_loss:.2f}) - poor convergence")
            elif val_loss > 5.0:
                issues.append(f"{variant.upper()}: High validation loss ({val_loss:.2f}) - may need more training")
        
        return issues
    
    def _check_configuration(self) -> List[str]:
        """Check for configuration issues."""
        issues = []
        config = self.data.get('config_common', {})
        
        # Check data size vs model capacity
        train_samples = config.get('train_limit', 0)
        d_model = config.get('d_model', 0)
        n_layers = config.get('n_layers', 0)
        
        if train_samples and d_model and n_layers:
            # Rough parameter estimate for transformer
            vocab_size = self.data.get('results', {}).get('dense', {}).get('vocab_size', 50257)
            approx_params = (
                vocab_size * d_model +  # embedding
                n_layers * (4 * d_model * d_model + 2 * d_model * config.get('ff_dim', d_model * 4)) +  # layers
                vocab_size * d_model  # output head (if not tied)
            )
            
            samples_per_param = train_samples / approx_params if approx_params > 0 else 0
            
            if samples_per_param < 0.001:  # Less than 1 sample per 1000 parameters
                issues.append(f"Model too large for dataset: {samples_per_param:.4f} samples/param (need >0.01)")
            elif samples_per_param < 0.01:
                issues.append(f"Model may be too large: {samples_per_param:.4f} samples/param (consider smaller model)")
        
        # Check regularization
        dropout = config.get('dropout', 0.0)
        weight_decay = config.get('weight_decay', 0.0)
        
        if train_samples and train_samples < 10000:  # Small dataset
            if dropout < 0.2:
                issues.append(f"Low dropout ({dropout}) for small dataset - consider ≥0.3")
            if weight_decay < 0.05:
                issues.append(f"Low weight decay ({weight_decay}) for small dataset - consider ≥0.1")
        
        # Check learning rate
        lr = config.get('lr', 0.0)
        if lr > 0.001:
            issues.append(f"High learning rate ({lr}) - consider ≤1e-3 for stability")
        
        return issues
    
    def _check_model_capacity(self) -> List[str]:
        """Check model capacity issues."""
        issues = []
        config = self.data.get('config_common', {})
        
        d_model = config.get('d_model', 0)
        n_heads = config.get('n_heads', 0)
        
        # Check if d_model is divisible by n_heads
        if d_model and n_heads and d_model % n_heads != 0:
            issues.append(f"d_model ({d_model}) not divisible by n_heads ({n_heads})")
        
        # Check for very small models that might underfit
        if d_model and d_model < 64:
            issues.append(f"Very small model (d_model={d_model}) - may underfit")
        
        return issues


def main():
    parser = argparse.ArgumentParser(description="Diagnose training results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"❌ File not found: {args.results_file}")
        sys.exit(1)
    
    diagnostics = TrainingDiagnostics(args.results_file)
    diagnostics.run_all_diagnostics()


if __name__ == "__main__":
    main()