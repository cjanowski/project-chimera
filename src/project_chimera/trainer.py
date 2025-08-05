from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils.device import get_device


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 200
    log_every: int = 20
    eval_every: int = 100
    grad_clip: float = 1.0
    amp: bool = True
    # Early stopping parameters
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    # Validation monitoring
    val_check_interval: int = 5  # Check validation every N steps
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", or "none"
    warmup_steps: int = 50
    # Label smoothing
    label_smoothing: float = 0.1


class BaselineTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Use new torch.amp API for CUDA only; disable on MPS/CPU
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and self.device.type == "cuda"))

        self._step = 0
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Validation history for monitoring
        self.val_loss_history = []
        
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.cfg.lr_scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.cfg.max_steps - self.cfg.warmup_steps,
                eta_min=self.cfg.lr * 0.1
            )
        elif self.cfg.lr_scheduler_type == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.cfg.max_steps - self.cfg.warmup_steps
            )
        else:
            return None
    
    def _warmup_lr(self):
        """Apply learning rate warmup."""
        if self._step < self.cfg.warmup_steps:
            warmup_factor = self._step / self.cfg.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.lr * warmup_factor
                
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early based on validation loss."""
        if val_loss < self.best_val_loss - self.cfg.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.cfg.early_stopping_patience:
                self.early_stopped = True
                return True
        return False
        
    def _label_smoothing_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss."""
        if self.cfg.label_smoothing == 0.0:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, logits.size(-1))  # [B*T, V]
        labels_flat = labels.view(-1)  # [B*T]
        
        # Create mask for valid tokens (not padding)
        valid_mask = labels_flat != -100
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Get valid logits and labels
        valid_logits = logits_flat[valid_mask]  # [N, V]
        valid_labels = labels_flat[valid_mask]  # [N]
        
        # Compute log probabilities
        log_probs = F.log_softmax(valid_logits, dim=-1)
        
        # One-hot encoding
        num_classes = valid_logits.size(-1)
        one_hot = torch.zeros_like(valid_logits).scatter_(1, valid_labels.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_labels = one_hot * (1 - self.cfg.label_smoothing) + \
                       self.cfg.label_smoothing / num_classes
        
        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        # Use new autocast API for CUDA only; MPS/CPU will run in FP32
        with torch.amp.autocast("cuda", enabled=(self.cfg.amp and self.device.type == "cuda")):
            # Get logits without loss computation from model
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = out["logits"]
            # Compute our custom loss with label smoothing
            loss = self._label_smoothing_loss(logits, labels)
            out["loss"] = loss
        return out

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        losses = []
        for batch in self.val_loader:
            out = self._forward_batch(batch)
            loss = out["loss"]
            losses.append(loss.item())
        self.model.train()
        avg_loss = sum(losses) / max(1, len(losses))
        return {"val_loss": avg_loss}

    def train(self) -> Dict[str, float]:
        self.model.train()
        pbar = tqdm(total=self.cfg.max_steps, desc="train", leave=False)
        running = 0.0
        
        while self._step < self.cfg.max_steps and not self.early_stopped:
            for batch in self.train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                out = self._forward_batch(batch)
                loss = out["loss"]

                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Apply learning rate scheduling
                if self._step < self.cfg.warmup_steps:
                    self._warmup_lr()
                elif self.scheduler is not None:
                    self.scheduler.step()

                self._step += 1
                pbar.update(1)
                running += loss.item()

                # Logging
                if self._step % self.cfg.log_every == 0:
                    avg = running / self.cfg.log_every
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{current_lr:.2e}"})
                    running = 0.0

                # Frequent validation checks for early stopping
                if self._step % self.cfg.val_check_interval == 0:
                    metrics = self.evaluate()
                    val_loss = metrics['val_loss']
                    self.val_loss_history.append(val_loss)
                    
                    # Check for early stopping
                    if self._check_early_stopping(val_loss):
                        pbar.set_postfix({
                            "loss": f"{loss.item():.4f}", 
                            "val_loss": f"{val_loss:.4f}",
                            "early_stop": "True"
                        })
                        print(f"\nEarly stopping triggered at step {self._step}. Best val_loss: {self.best_val_loss:.4f}")
                        break
                    
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}", 
                        "val_loss": f"{val_loss:.4f}",
                        "patience": f"{self.patience_counter}/{self.cfg.early_stopping_patience}"
                    })

                # Regular evaluation (less frequent than validation checks)
                elif self._step % self.cfg.eval_every == 0:
                    metrics = self.evaluate()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "val_loss": f"{metrics['val_loss']:.4f}"})

                if self._step >= self.cfg.max_steps:
                    break
                    
            if self.early_stopped:
                break
                
        pbar.close()
        final_metrics = self.evaluate()
        final_metrics['early_stopped'] = self.early_stopped
        final_metrics['best_val_loss'] = self.best_val_loss
        final_metrics['final_step'] = self._step
        return final_metrics