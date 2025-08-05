# Troubleshooting Guide

This document captures common issues encountered during Project Chimera development and their solutions.

## Table of Contents
- [Training Issues](#training-issues)
  - [Zero Loss Problem (Severe Overfitting)](#zero-loss-problem-severe-overfitting)
- [Data Issues](#data-issues)
- [Model Architecture Issues](#model-architecture-issues)
- [System Issues](#system-issues)
- [Apple Silicon (M1/M2/M3) MPS Tuning and Warnings](#apple-silicon-m1m2m3-mps-tuning-and-warnings)

---

## Training Issues

### Zero Loss Problem (Severe Overfitting)

**Problem**: Model achieves 0.0000 loss on both training and validation data, indicating perfect memorization rather than learning.

#### Symptoms
- Training loss drops to exactly 0.0 within a few hundred steps
- Validation loss also reaches 0.0 
- Model appears to be "learning perfectly" but is actually memorizing
- Poor generalization to new data

#### Root Cause Analysis

In our case study, the issue was caused by:

1. **Severe Model-to-Data Ratio Imbalance**
   - Training samples: 2,048
   - Model parameters: ~2.8M (4-layer transformer, 256 dim, 4 heads)
   - Ratio: 1,367 parameters per training sample

2. **Insufficient Regularization**
   - Dropout: 0.1 (too low for the model size)
   - Weight decay: 0.01 (insufficient)
   - No early stopping mechanism

3. **Extended Training**
   - 5,000 training steps on small dataset
   - No validation monitoring during training
   - No stopping criteria based on generalization

#### Evidence Found

From `runs/phase8_long_dense_dense.json:39`:
```json
"final_metrics": {
  "val_loss": 0.0
}
```

**Key Diagnostic**: Both training AND validation loss = 0.0 confirms overfitting rather than data leakage.

#### Solution Implementation

We implemented a comprehensive overfitting mitigation strategy:

**1. Model Architecture Changes** (`scripts/train_agnews.py`)
```python
# Reduced model capacity
d_model: int = 128        # was 256 (50% reduction)
n_layers: int = 2         # was 4 (50% reduction)  
ff_dim: int = 512         # was 1024 (50% reduction)
```

**2. Enhanced Regularization**
```python
dropout: float = 0.4      # was 0.1 (4x increase)
weight_decay: float = 0.1 # was 0.01 (10x increase)
lr: float = 1e-4          # was 3e-4 (reduced for stability)
```

**3. Training Loop Improvements** (`src/project_chimera/trainer.py`)
- **Early Stopping**: Added patience-based monitoring with 5-step patience
- **Frequent Validation**: Check validation loss every 20 steps instead of 500
- **Label Smoothing**: Implemented smoothing factor of 0.1 to prevent overconfident predictions
- **Learning Rate Scheduling**: Added warmup and decay for stable training

**4. Data Augmentation** (`src/project_chimera/data/preprocess.py`)
- **Synonym Replacement**: Random word substitution using NLTK
- **Word Dropout**: Randomly remove words with 10% probability
- **Word Shuffling**: Randomly shuffle word order with 15% probability
- Applied only to training data to increase effective dataset size

**5. Enhanced Monitoring**
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
```

#### Expected Results After Fix

- **Training loss > 0.1**: Eliminates perfect memorization
- **Validation loss tracks training loss**: Indicates proper generalization
- **Automatic early stopping**: Prevents overfitting before it occurs
- **Stable convergence**: Smoother loss curves

#### Testing the Fix

Run the improved training:
```bash
python scripts/train_agnews.py --run_dense
```

Monitor for these improvements:
- Training loss staying above 0.1
- Validation loss following training loss trends  
- Early stopping triggering if needed
- Better model generalization

#### Prevention Strategies

1. **Always check model-to-data ratio**
   - Rule of thumb: 10-100 samples per parameter for good generalization
   - For 2.8M parameters, need at least 28K-280K samples

2. **Implement validation monitoring from start**
   - Check validation loss frequently (every 10-50 steps)
   - Plot training vs validation curves
   - Stop when validation loss stops improving

3. **Use strong regularization for small datasets**
   - Dropout ≥ 0.3 for transformer models on small data
   - Weight decay ≥ 0.01, potentially up to 0.1
   - Consider additional techniques like label smoothing

4. **Start with smaller models**
   - Begin with minimal viable architecture
   - Scale up only if underfitting occurs
   - Better to underfit slightly than severely overfit

#### Key Takeaways

- **Zero loss is always a red flag** - models should not achieve perfect accuracy on real data
- **Validation loss is crucial** - it's your canary in the coal mine for overfitting
- **Model capacity must match data size** - more parameters require more data
- **Regularization is not optional** - especially critical for small datasets

---

## Data Issues

### Data Leakage
*To be documented as issues arise*

### Tokenization Problems  
*To be documented as issues arise*

---

## Model Architecture Issues

### Memory Issues
*To be documented as issues arise*

### NaN Gradients
*To be documented as issues arise*

---

## System Issues

### Device Compatibility
*To be documented as issues arise*

### Environment Setup
*To be documented as issues arise*

### Apple Silicon (M1/M2/M3) MPS Tuning and Warnings

This section documents changes we made to improve ergonomics and performance when running on Apple Silicon using the MPS backend, without altering the overall research flow.

#### Symptoms

- HuggingFace tokenizers warning about fork/parallelism:
  “The current process just got forked… Disabling parallelism to avoid deadlocks…”
- PyTorch warning on DataLoader:
  "'pin_memory' argument is set as true but not supported on MPS now…"
- Deprecated AMP warnings:
  - torch.cuda.amp.GradScaler(...) is deprecated
  - torch.cuda.amp.autocast(...) is deprecated

#### Root Cause

- Tokenizers library uses parallelism that can conflict with multi-processing on macOS when processes are forked after tokenizers have been used.
- PyTorch pin_memory is CUDA-specific; MPS currently does not benefit and warns if enabled.
- Newer PyTorch prefers torch.amp APIs over torch.cuda.amp and restricts autocast usage to the actual device type (CUDA only in our case). On MPS, AMP is not active and we should run FP32.

#### Fix Summary

1) Disable tokenizers parallelism and prefer higher matmul precision
   - File: [scripts/train_agnews.py:23-29](scripts/train_agnews.py:23)
   - Changes:
     - Set environment variable TOKENIZERS_PARALLELISM=false to avoid fork-related warnings
     - Call torch.set_float32_matmul_precision("high") to enable potentially faster matmul kernels on MPS/CPU

2) Avoid pin_memory on MPS in DataLoaders
   - File: [scripts/train_agnews.py:85-96](scripts/train_agnews.py:85)
   - Changes:
     - effective_pin_memory = cfg.pin_memory and (device.type != "mps")
     - Pass effective_pin_memory into build_dataloaders(...)

3) Modernize AMP usage for CUDA while keeping MPS in FP32
   - File: [src/project_chimera/trainer.py:51-53](src/project_chimera/trainer.py:51), [src/project_chimera/trainer.py:143-151](src/project_chimera/trainer.py:143)
   - Changes:
     - Replace torch.cuda.amp.GradScaler(...) with torch.amp.GradScaler("cuda", enabled=(...))
     - Replace torch.cuda.amp.autocast(...) with torch.amp.autocast("cuda", enabled=(...))
   - Effect:
     - On CUDA: AMP works via the new API
     - On MPS: AMP disabled, runs FP32 cleanly without deprecation warnings

4) Package baselining for macOS M3
   - Installed/updated key packages (completed successfully via pip):
     - torch/torchvision/torchaudio (CPU/MPS wheels)
     - transformers==4.42.4, datasets==2.20.0, tokenizers==0.19.1
     - accelerate==0.33.0, evaluate==0.4.1, sentencepiece, safetensors

#### Verification

- Tokenizers fork/parallelism warning no longer appears at script start.
- pin_memory warning no longer appears when using MPS device.
- AMP deprecation warnings removed; training/logging behaves as before.
- No changes were made to dataset content or experiment logic beyond these ergonomics.

#### References (Code Locations)

- Env and matmul precision setup:
  - [scripts/train_agnews.py:23](scripts/train_agnews.py:23)
- MPS-aware pin_memory:
  - [scripts/train_agnews.py:85](scripts/train_agnews.py:85)
- AMP modernization (CUDA-only enablement):
  - [src/project_chimera/trainer.py:51](src/project_chimera/trainer.py:51)
  - [src/project_chimera/trainer.py:143](src/project_chimera/trainer.py:143)

#### Notes

- Autocast remains disabled on MPS; if PyTorch adds stable MPS AMP, we can revisit.
- For deterministic comparisons, keep the same seeds and ensure identical data limits across dense vs MoE runs.
- If needed, we can export requirements with pinned versions used for these runs for exact reproducibility.

---

## Contributing to This Guide

When you encounter a new issue:

1. **Document the problem clearly**
   - What were you trying to do?
   - What happened instead?
   - What error messages did you see?

2. **Include diagnostic information**
   - Relevant code snippets
   - Configuration values
   - Log outputs or metrics

3. **Explain the root cause**
   - Why did this happen?
   - What conditions led to the issue?

4. **Detail the solution**
   - What changes were made?
   - Why does this fix work?
   - How to verify the fix worked?

5. **Add prevention strategies**
   - How to avoid this in the future?
   - What to watch out for?
   - Best practices to follow?

This living document helps the research community learn from our experiences and build more robust training pipelines.