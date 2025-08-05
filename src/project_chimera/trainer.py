from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import torch
from torch import nn, optim
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
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and self.device.type == "cuda"))

        self._step = 0

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(self.cfg.amp and self.device.type == "cuda")):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
        while self._step < self.cfg.max_steps:
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

                self._step += 1
                pbar.update(1)
                running += loss.item()

                if self._step % self.cfg.log_every == 0:
                    avg = running / self.cfg.log_every
                    pbar.set_postfix({"loss": f"{avg:.4f}"})
                    running = 0.0

                if self._step % self.cfg.eval_every == 0:
                    metrics = self.evaluate()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "val_loss": f"{metrics['val_loss']:.4f}"})

                if self._step >= self.cfg.max_steps:
                    break
        pbar.close()
        final_metrics = self.evaluate()
        return final_metrics