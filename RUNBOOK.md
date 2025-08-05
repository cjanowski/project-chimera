# Project Chimera Runbook

Quick reference guide for common operations, commands, and workflows.

## Table of Contents
- [Quick Start](#quick-start)
- [Training Commands](#training-commands)
- [Diagnostic Tools](#diagnostic-tools)
- [Data Management](#data-management)
- [Testing & Validation](#testing--validation)
- [Troubleshooting Scripts](#troubleshooting-scripts)

---

## Quick Start

### Environment Setup
```bash
# Install dependencies
pip install -e .[dev]

# Run tests to verify setup
pytest -q

# Check GPU/device availability
python -c "from src.project_chimera.utils.device import get_device, device_name; print(f'Device: {device_name(get_device())}')"
```

### Quick Smoke Test
```bash
# Run minimal training test (2 minutes)
python scripts/train_agnews.py \
  --train_limit 128 \
  --val_limit 32 \
  --max_steps 20 \
  --batch_size 8 \
  --run_dense
```

---

## Training Commands

### Standard Training Runs

#### Dense Baseline (Recommended Settings)
```bash
# Balanced configuration - good starting point
python scripts/train_agnews.py \
  --run_dense \
  --train_limit 20000 \
  --val_limit 5000 \
  --batch_size 32 \
  --d_model 128 \
  --n_layers 2 \
  --ff_dim 512 \
  --dropout 0.4 \
  --weight_decay 0.1 \
  --lr 1e-4 \
  --max_steps 1000 \
  --log_every 50 \
  --eval_every 100
```

#### MoE Variant
```bash
# MoE with 4 experts, top-1 routing
python scripts/train_agnews.py \
  --run_moe \
  --train_limit 20000 \
  --val_limit 5000 \
  --batch_size 32 \
  --d_model 128 \
  --n_layers 2 \
  --ff_dim 512 \
  --dropout 0.4 \
  --weight_decay 0.1 \
  --lr 1e-4 \
  --max_steps 1000 \
  --moe_n_experts 4 \
  --moe_top_k 1
```

#### Comparative Run (Both Models)
```bash
# Train both dense and MoE for comparison
python scripts/train_agnews.py \
  --run_dense \
  --run_moe \
  --train_limit 10000 \
  --val_limit 2500 \
  --batch_size 16 \
  --max_steps 500
```

### Hyperparameter Sweeps

#### Model Size Ablation
```bash
# Small model
python scripts/train_agnews.py --run_dense --d_model 64 --n_layers 2 --ff_dim 256 --run_tag "small"

# Medium model  
python scripts/train_agnews.py --run_dense --d_model 128 --n_layers 2 --ff_dim 512 --run_tag "medium"

# Large model
python scripts/train_agnews.py --run_dense --d_model 256 --n_layers 4 --ff_dim 1024 --run_tag "large"
```

#### Regularization Sweep
```bash
# Low regularization
python scripts/train_agnews.py --run_dense --dropout 0.1 --weight_decay 0.01 --run_tag "low_reg"

# High regularization  
python scripts/train_agnews.py --run_dense --dropout 0.5 --weight_decay 0.2 --run_tag "high_reg"
```

### Long Training Runs
```bash
# Extended training for convergence study
python scripts/train_agnews.py \
  --run_dense \
  --train_limit 50000 \
  --val_limit 10000 \
  --max_steps 5000 \
  --eval_every 200 \
  --run_tag "extended"
```

---

## Diagnostic Tools

### Training Diagnostics Script
See `scripts/diagnose_training.py` (created below) for automated issue detection.

```bash
# Run full training diagnostics
python scripts/diagnose_training.py runs/your_experiment.json

# Quick diagnostic on latest run
python scripts/diagnose_training.py $(ls -t runs/*.json | head -1)
```

### Manual Diagnostics

#### Check Results Files
```bash
# List all training runs
ls -la runs/

# View latest results
cat $(ls -t runs/*.json | head -1) | jq '.results.dense.final_metrics'

# Compare multiple runs
for f in runs/*.json; do
  echo "=== $f ==="
  jq -r '.results.dense.final_metrics.val_loss // "N/A"' "$f"
done
```

#### Model Parameter Count
```bash
# Get model size for different configurations
python -c "
from src.project_chimera.baseline import BaselineConfig, GPTDecoder
configs = [
  ('small', dict(vocab_size=50257, d_model=64, n_layers=2, ff_dim=256)),
  ('medium', dict(vocab_size=50257, d_model=128, n_layers=2, ff_dim=512)),
  ('large', dict(vocab_size=50257, d_model=256, n_layers=4, ff_dim=1024))
]
for name, cfg in configs:
  model = GPTDecoder(BaselineConfig(**cfg))
  params = sum(p.numel() for p in model.parameters())
  print(f'{name}: {params:,} parameters')
"
```

#### Data Statistics
```bash
# Check tokenizer vocab size
python -c "
from src.project_chimera.data.preprocess import build_tokenizer, TokenizerConfig
tok = build_tokenizer(TokenizerConfig())
print(f'Vocab size: {tok.vocab_size}')
print(f'Pad token: {tok.pad_token} (id: {tok.pad_token_id})')
print(f'EOS token: {tok.eos_token} (id: {tok.eos_token_id})')
"

# Sample data inspection
python -c "
from src.project_chimera.data.preprocess import CausalTextDataset, TokenizerConfig, build_tokenizer
tok = build_tokenizer(TokenizerConfig())
ds = CausalTextDataset('train', tok, TokenizerConfig(), limit=5)
print(f'Dataset size: {len(ds)}')
for i in range(min(3, len(ds))):
  item = ds[i]
  print(f'Sample {i}: input_ids shape {item[\"input_ids\"].shape}')
  print(f'  Text: {tok.decode(item[\"input_ids\"], skip_special_tokens=True)[:100]}...')
"
```

---

## Data Management

### Download Dataset
```bash
# Download AG News dataset
python scripts/download_ag_news.py

# Verify download
ls -la data/ag_news/
```

### Data Preprocessing Test
```bash
# Test data pipeline
python -c "
from src.project_chimera.data.preprocess import build_dataloaders, build_tokenizer, TokenizerConfig
tok = build_tokenizer(TokenizerConfig())
train_loader, val_loader = build_dataloaders(tok, TokenizerConfig(), train_limit=100, val_limit=50, batch_size=4)
print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
for batch in train_loader:
  print(f'Batch shape: {batch[\"input_ids\"].shape}')
  break
"
```

---

## Testing & Validation

### Unit Tests
```bash
# Run all tests
pytest -v

# Run specific test categories
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_device_and_seed.py -v

# Test with coverage
pytest --cov=src/project_chimera --cov-report=html
```

### Integration Tests
```bash
# End-to-end training test (fast)
python scripts/train_agnews.py \
  --train_limit 64 \
  --val_limit 32 \
  --max_steps 5 \
  --batch_size 4 \
  --run_dense \
  --run_tag "integration_test"
```

### Model Validation
```bash
# Test model forward pass
python -c "
import torch
from src.project_chimera.baseline import BaselineConfig, GPTDecoder
cfg = BaselineConfig(vocab_size=1000, d_model=64, n_layers=2)
model = GPTDecoder(cfg)
x = torch.randint(0, 1000, (2, 10))
out = model(x, labels=x)
print(f'Loss: {out[\"loss\"].item():.4f}')
print(f'Logits shape: {out[\"logits\"].shape}')
assert out[\"loss\"].item() > 0, 'Loss should be > 0 for random inputs'
print('✓ Model validation passed')
"
```

---

## Troubleshooting Scripts

### Quick Health Check
```bash
# Run comprehensive health check
python scripts/health_check.py
```

### Common Issues

#### Check for Overfitting
```bash
# Analyze training results for overfitting signs
python -c "
import json
import sys
try:
  with open(sys.argv[1]) as f:
    data = json.load(f)
  val_loss = data['results']['dense']['final_metrics']['val_loss']
  if val_loss < 0.01:
    print('⚠️  WARNING: Very low validation loss ({:.6f}) - possible overfitting'.format(val_loss))
  elif val_loss < 0.1:
    print('ℹ️  INFO: Low validation loss ({:.4f}) - monitor for overfitting'.format(val_loss))
  else:
    print('✓ Validation loss looks healthy: {:.4f}'.format(val_loss))
except Exception as e:
  print(f'❌ Error reading results: {e}')
" runs/your_experiment.json
```

#### Device Issues
```bash
# Test device compatibility
python -c "
from src.project_chimera.utils.device import get_device, device_name
import torch
device = get_device()
print(f'Using device: {device_name(device)}')
if device.type == 'cuda':
  print(f'CUDA version: {torch.version.cuda}')
  print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
elif device.type == 'mps':
  print('Metal Performance Shaders available')
else:
  print('Using CPU')
"
```

### Performance Benchmarking
```bash
# Quick performance test
python -c "
import time
import torch
from src.project_chimera.baseline import BaselineConfig, GPTDecoder
from src.project_chimera.utils.device import get_device

device = get_device()
cfg = BaselineConfig(vocab_size=50257, d_model=128, n_layers=2, ff_dim=512)
model = GPTDecoder(cfg).to(device)
x = torch.randint(0, 50257, (16, 128)).to(device)

# Warmup
for _ in range(10):
  _ = model(x)

# Benchmark
torch.cuda.synchronize() if device.type == 'cuda' else None
start = time.time()
for _ in range(100):
  _ = model(x)
torch.cuda.synchronize() if device.type == 'cuda' else None
end = time.time()

print(f'Inference: {(end-start)*10:.1f}ms per forward pass')
print(f'Throughput: {16*128*100/(end-start):.0f} tokens/sec')
"
```

---

## Batch Operations

### Multiple Experiments
```bash
# Run parameter sweep
for d_model in 64 128 256; do
  for n_layers in 2 4; do
    echo "Training d_model=$d_model n_layers=$n_layers"
    python scripts/train_agnews.py \
      --run_dense \
      --d_model $d_model \
      --n_layers $n_layers \
      --max_steps 200 \
      --run_tag "sweep_${d_model}_${n_layers}"
  done
done
```

### Results Analysis
```bash
# Compare all runs
echo "Run,Val_Loss,Model_Size"
for f in runs/*.json; do
  run=$(basename "$f" .json)
  val_loss=$(jq -r '.results.dense.final_metrics.val_loss // "N/A"' "$f")
  d_model=$(jq -r '.config_common.d_model // "N/A"' "$f")
  n_layers=$(jq -r '.config_common.n_layers // "N/A"' "$f")
  echo "$run,$val_loss,${d_model}x${n_layers}"
done
```

### Cleanup
```bash
# Remove old experiment files
find runs/ -name "*.json" -mtime +7 -delete

# Archive completed experiments
mkdir -p archive/$(date +%Y-%m)
mv runs/*.json archive/$(date +%Y-%m)/
```

---

## Environment Variables

### Useful Settings
```bash
# Disable tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# PyTorch settings
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Apple Silicon
export OMP_NUM_THREADS=4              # CPU parallelism

# Debugging
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1         # For CUDA debugging
```

### Development Mode
```bash
# Development settings
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled  # If using wandb later
```

---

## Automation Examples

### Training Pipeline Script
```bash
#!/bin/bash
# training_pipeline.sh

set -e

echo "🚀 Starting training pipeline..."

# Health check
python scripts/health_check.py

# Download data if needed
if [ ! -d "data/ag_news" ]; then
  echo "📥 Downloading dataset..."
  python scripts/download_ag_news.py
fi

# Run experiments
echo "🔬 Running experiments..."
python scripts/train_agnews.py --run_dense --run_tag "pipeline_$(date +%Y%m%d_%H%M)"

# Diagnose results
echo "🔍 Running diagnostics..."
latest_run=$(ls -t runs/*.json | head -1)
python scripts/diagnose_training.py "$latest_run"

echo "✅ Pipeline complete!"
```

### Monitoring Script
```bash
#!/bin/bash
# monitor_training.sh

# Monitor training progress in real-time
watch -n 5 'ls -t runs/*.json | head -5 | xargs -I {} sh -c "echo {}; jq -r \".results.dense.final_metrics.val_loss // \\\"Running...\\\"\" {}"'
```

Make scripts executable:
```bash
chmod +x scripts/training_pipeline.sh scripts/monitor_training.sh
```

---

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).