#!/usr/bin/env python3
"""
Project Chimera Quick Start

Interactive setup and testing script for new users or fresh environments.
Handles environment verification, data download, and runs a basic experiment.

Usage:
    python scripts/quick_start.py
    python scripts/quick_start.py --skip-download  # Skip data download
    python scripts/quick_start.py --minimal        # Minimal test only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command with nice output formatting."""
    print(f"🔧 {description}...")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            # Indent output
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Command failed with exit code {e.returncode}")
        if e.stderr:
            for line in e.stderr.strip().split('\n'):
                print(f"   ❌ {line}")
        return False


def check_python_version():
    """Verify Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def setup_environment():
    """Install dependencies and set up environment."""
    print("\n📦 Setting up environment...")
    
    # Check if we're in the right directory
    if not Path("setup.py").exists() and not Path("pyproject.toml").exists():
        print("   ❌ No setup.py or pyproject.toml found")
        print("   💡 Run this script from the project root directory")
        return False
    
    # Install in development mode
    success = run_command("pip install -e .[dev]", "Installing dependencies")
    if not success:
        print("   💡 Try: pip install --upgrade pip")
        return False
    
    return True


def download_data(skip=False):
    """Download required datasets."""
    if skip:
        print("\n📁 Skipping data download")
        return True
    
    print("\n📁 Downloading data...")
    
    # Check if data already exists
    data_dir = Path("data/ag_news")
    if data_dir.exists() and any(data_dir.iterdir()):
        print("   ℹ️  Data already exists, skipping download")
        return True
    
    # Check if download script exists
    download_script = Path("scripts/download_ag_news.py")
    if not download_script.exists():
        print("   ⚠️  Download script not found, skipping")
        return True
    
    return run_command("python scripts/download_ag_news.py", "Downloading AG News dataset")


def run_health_check():
    """Run comprehensive health check."""
    print("\n🏥 Running health check...")
    return run_command("python scripts/health_check.py --quick", "System health check")


def run_minimal_experiment():
    """Run a minimal training experiment."""
    print("\n🧪 Running minimal experiment...")
    
    cmd = """python scripts/train_agnews.py \
  --run_dense \
  --train_limit 128 \
  --val_limit 32 \
  --max_steps 10 \
  --batch_size 4 \
  --d_model 64 \
  --n_layers 1 \
  --run_tag "quick_start_test"
    """.replace('\n', ' ').strip()
    
    return run_command(cmd, "Running minimal training test")


def run_diagnostic_on_results():
    """Run diagnostics on the experiment results."""
    print("\n🔍 Analyzing results...")
    
    # Find the latest results file
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("   ❌ No runs directory found")
        return False
    
    json_files = list(runs_dir.glob("*.json"))
    if not json_files:
        print("   ❌ No result files found")
        return False
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    return run_command(f"python scripts/diagnose_training.py {latest_file}", "Analyzing training results")


def print_next_steps():
    """Print guidance for next steps."""
    print("\n" + "🎯 Next Steps".center(50, "="))
    print("Your Project Chimera setup is complete! Here's what you can do:")
    print()
    print("📚 Learn more:")
    print("   • Read the README.md for full documentation")
    print("   • Check RUNBOOK.md for common commands")
    print("   • See TROUBLESHOOTING.md if you hit issues")
    print()
    print("🚀 Run experiments:")
    print("   • python scripts/train_agnews.py --run_dense")
    print("   • python scripts/train_agnews.py --run_moe")
    print("   • python scripts/train_agnews.py --run_dense --run_moe")
    print()
    print("🔧 Development:")
    print("   • pytest -v                    # Run tests")
    print("   • python scripts/health_check.py  # Check system")
    print("   • ls runs/                     # View results")
    print()
    print("💡 Tips:")
    print("   • Start with small experiments to test the pipeline")
    print("   • Monitor validation loss to catch overfitting early")
    print("   • Use the diagnostic script to analyze results")
    print()


def main():
    parser = argparse.ArgumentParser(description="Project Chimera Quick Start")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--minimal", action="store_true", help="Minimal setup only (no experiment)")
    parser.add_argument("--no-install", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print("🚀 Project Chimera Quick Start")
    print("=" * 50)
    
    steps = [
        ("Python Version", check_python_version),
        ("Environment Setup", lambda: setup_environment() if not args.no_install else True),
        ("Data Download", lambda: download_data(args.skip_download)),
        ("Health Check", run_health_check),
    ]
    
    if not args.minimal:
        steps.extend([
            ("Minimal Experiment", run_minimal_experiment),
            ("Result Analysis", run_diagnostic_on_results),
        ])
    
    # Run all setup steps
    failed_steps = []
    for step_name, step_func in steps:
        try:
            if not step_func():
                failed_steps.append(step_name)
        except KeyboardInterrupt:
            print(f"\n⏹️  Setup interrupted during {step_name}")
            sys.exit(1)
        except Exception as e:
            print(f"\n💥 Unexpected error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    if not failed_steps:
        print("🎉 Quick start completed successfully!")
        if not args.minimal:
            print_next_steps()
    else:
        print(f"⚠️  Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"   ❌ {step}")
        print("\n💡 Check the error messages above and refer to TROUBLESHOOTING.md")
        
        # Still show next steps if most things worked
        if len(failed_steps) <= 2:
            print_next_steps()


if __name__ == "__main__":
    main()