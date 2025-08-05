from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

# Use the parquet-backed AG News dataset utilities already present
# We will wrap them for causal LM style inputs.
from .ag_news import (
    TokenizerConfig as ParquetTokCfg,   # to avoid confusion with ours
    get_tokenizer as get_hf_tokenizer,  # not used directly here
    ParquetTextClassificationDataset,
    build_datasets_and_loaders,
)
from pathlib import Path

DEFAULT_MODEL_NAME = "gpt2"


@dataclass
class TokenizerConfig:
    pretrained_name: str = DEFAULT_MODEL_NAME
    lowercase: bool = True
    max_length: int = 128
    padding: str = "max_length"  # "max_length" or "longest"
    truncation: bool = True
    add_special_tokens: bool = True
    # Data augmentation settings
    augment_data: bool = True
    augment_prob: float = 0.3  # Probability of applying augmentation
    max_augmentations_per_text: int = 2  # Maximum number of different augmentations per text


def build_tokenizer(cfg: TokenizerConfig):
    tok = AutoTokenizer.from_pretrained(cfg.pretrained_name)
    # GPT-2 has no pad token by default. Set pad_token to eos_token for batching.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def apply_text_augmentation(text: str, cfg: TokenizerConfig) -> List[str]:
    """Apply various text augmentation techniques to increase dataset diversity."""
    if not cfg.augment_data:
        return [text]
    
    augmented_texts = [text]  # Always include original
    
    # Define augmentation functions
    def synonym_replacement(text: str) -> str:
        """Simple synonym replacement using common word substitutions."""
        substitutions = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely'],
            'new': ['recent', 'fresh', 'latest', 'novel'],
            'old': ['ancient', 'vintage', 'aged'],
            'important': ['crucial', 'vital', 'significant', 'key'],
            'said': ['stated', 'mentioned', 'noted', 'declared'],
            'company': ['firm', 'corporation', 'business', 'enterprise'],
            'people': ['individuals', 'persons', 'citizens', 'folks'],
            'market': ['marketplace', 'trading', 'economy'],
            'government': ['administration', 'authority', 'officials'],
        }
        
        words = text.split()
        new_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in substitutions and random.random() < 0.3:
                replacement = random.choice(substitutions[clean_word])
                # Preserve original case pattern
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(word.replace(clean_word, replacement, 1))
            else:
                new_words.append(word)
        return ' '.join(new_words)
    
    def word_dropout(text: str, drop_rate: float = 0.1) -> str:
        """Randomly drop words to create variations."""
        words = text.split()
        if len(words) <= 3:  # Don't drop words from very short texts
            return text
        
        keep_words = []
        for word in words:
            if random.random() > drop_rate:
                keep_words.append(word)
        
        # Ensure we keep at least half the words
        if len(keep_words) < len(words) // 2:
            return text
        
        return ' '.join(keep_words)
    
    def word_shuffle(text: str, shuffle_rate: float = 0.2) -> str:
        """Randomly shuffle some words within sentences."""
        sentences = text.split('.')
        shuffled_sentences = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) <= 3:
                shuffled_sentences.append(sentence)
                continue
                
            # Randomly shuffle a portion of words
            num_to_shuffle = max(1, int(len(words) * shuffle_rate))
            indices_to_shuffle = random.sample(range(len(words)), min(num_to_shuffle, len(words)))
            
            if len(indices_to_shuffle) >= 2:
                shuffled_indices = indices_to_shuffle.copy()
                random.shuffle(shuffled_indices)
                
                new_words = words.copy()
                for i, j in zip(indices_to_shuffle, shuffled_indices):
                    new_words[i] = words[j]
                
                shuffled_sentences.append(' '.join(new_words))
            else:
                shuffled_sentences.append(sentence)
        
        return '.'.join(shuffled_sentences)
    
    # Available augmentation techniques
    augmentation_functions = [
        lambda t: synonym_replacement(t),
        lambda t: word_dropout(t, 0.1),
        lambda t: word_shuffle(t, 0.15),
    ]
    
    # Apply random augmentations
    num_augmentations = min(cfg.max_augmentations_per_text, len(augmentation_functions))
    selected_augs = random.sample(augmentation_functions, num_augmentations)
    
    for aug_func in selected_augs:
        if random.random() < cfg.augment_prob:
            try:
                augmented = aug_func(text)
                if augmented != text and len(augmented.strip()) > 10:  # Avoid empty or too short texts
                    augmented_texts.append(augmented)
            except Exception:
                # Skip augmentation if it fails
                continue
    
    return augmented_texts


class CausalTextDataset(Dataset):
    """
    Produces (input_ids, attention_mask, labels) triples for causal LM from
    parquet-backed AG News shards by reading 'text' field and ignoring class labels.
    Lowercasing is applied if configured (before tokenization).
    """

    def __init__(
        self,
        split: str,
        tokenizer,
        cfg: TokenizerConfig,
        data_root: str | Path = "data/ag_news",
        limit: Optional[int] = None,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.cfg = cfg

        # Use ParquetTextClassificationDataset to read texts
        split_dir = Path(data_root) / {"train": "train", "test": "test", "validation": "validation"}.get(split, split)

        texts = []
        # Prefer parquet dataset if available; otherwise fall back to a tiny synthetic corpus so unit tests can run without data present.
        try:
            src_ds = ParquetTextClassificationDataset(split_dir)
            n = len(src_ds) if limit is None else min(limit, len(src_ds))
            for i in range(n):
                item = src_ds[i]
                t = str(item["text"])
                if cfg.lowercase:
                    t = t.lower()
                
                # Apply data augmentation only to training split
                if split == "train" and cfg.augment_data:
                    augmented_texts = apply_text_augmentation(t, cfg)
                    texts.extend(augmented_texts)
                else:
                    texts.append(t)
        except FileNotFoundError:
            # Fallback tiny samples for tests without requiring downloads
            tiny = [
                "Breaking news: market sees gains as tech leads rally.",
                "Sports update: local team secures championship title.",
                "Weather today: sunny skies expected across the region.",
                "Politics: new policy initiative announced by officials.",
                "Science report: researchers discover new exoplanet.",
                "Economy: inflation shows signs of cooling this quarter.",
                "Technology: startup unveils innovative AI-powered tool.",
                "Health: study links sleep quality with cognitive performance.",
            ]
            base_n = len(tiny) if limit is None else min(limit, len(tiny))
            for i in range(base_n):
                t = tiny[i % len(tiny)]
                if cfg.lowercase:
                    t = t.lower()
                
                # Apply data augmentation to fallback data as well for training
                if split == "train" and cfg.augment_data:
                    augmented_texts = apply_text_augmentation(t, cfg)
                    texts.extend(augmented_texts)
                else:
                    texts.append(t)

        enc = tokenizer(
            texts,
            max_length=cfg.max_length,
            padding=cfg.padding,
            truncation=cfg.truncation,
            add_special_tokens=cfg.add_special_tokens,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        # For causal LM, labels are input_ids with padding ignored
        self.labels = self.input_ids.clone()
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            self.labels[self.input_ids == pad_id] = -100

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def build_dataloaders(
    tokenizer,
    cfg: TokenizerConfig,
    train_limit: int = 2048,
    val_limit: int = 512,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
    data_root: str | Path = "data/ag_news",
) -> Tuple[DataLoader, DataLoader]:
    train_ds = CausalTextDataset("train", tokenizer, cfg, data_root=data_root, limit=train_limit)
    val_ds = CausalTextDataset("test", tokenizer, cfg, data_root=data_root, limit=val_limit)
    collate_fn = None  # already tensorized

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader