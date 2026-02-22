import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CheckpointMeta:
    step: int
    vocab_size: int
    embed_dim: int


class GPTModel(nn.Module):
    """Tiny GPT-like model, torchified.

    Current architecture mirrors your NumPy version:
    token ids -> token embedding -> (prefix-average causal "attention") -> lm_head -> logits

    Notes:
    - No manual backward: use autograd (loss.backward()).
    - No custom optimizer: create torch.optim.* in train.py.
    - Supports CPU/GPU via `device`.
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 32, device: Optional[str] = None):
        super().__init__()
        self.tokenizer = MyTokenizer()

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

        # Device handling: default to cuda if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # token embedding and projection head
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # init scaling similar to your numpy * 0.02
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: LongTensor [B, T] -> logits: FloatTensor [B, T, V]"""
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long, device=self.device)
        else:
            x = x.to(self.device)
        if x.dtype != torch.long:
            x = x.long()

        h = self.embedding(x)  # [B, T, D]
        h = self.prefix_average(h)  # [B, T, D]
        logits = self.lm_head(h)  # [B, T, V]
        return logits

    @staticmethod
    def prefix_average(h: torch.Tensor) -> torch.Tensor:
        """Prefix-average causal aggregation.

        h: [B, T, D]
        out[:, t, :] = mean_{k<=t} h[:, k, :]
        """
        # cumulative sum over time
        csum = torch.cumsum(h, dim=1)  # [B, T, D]
        T = h.size(1)
        denom = torch.arange(1, T + 1, device=h.device, dtype=h.dtype).view(1, T, 1)
        return csum / denom

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over all positions.

        logits: [B, T, V]
        targets: [B, T] (Long)
        """
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
        else:
            targets = targets.to(self.device)
        if targets.dtype != torch.long:
            targets = targets.long()

        B, T, V = logits.shape
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    def get_batch(
        self,
        tokens: List[int],
        batch_size: int,
        block_size: int,
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample (x, y) from a flat token list.

        Returns:
          x: LongTensor [B, T]
          y: LongTensor [B, T]
        """
        if device is None:
            device = str(self.device)
        dev = torch.device(device)

        n = len(tokens)
        if n < block_size + 2:
            raise ValueError(f"tokens too short: n={n}, block_size={block_size}")

        # sample random starts on CPU, then move to device
        starts = torch.randint(0, n - block_size - 1, (batch_size,), dtype=torch.long)
        x = torch.stack([torch.tensor(tokens[s : s + block_size], dtype=torch.long) for s in starts], dim=0)
        y = torch.stack([torch.tensor(tokens[s + 1 : s + block_size + 1], dtype=torch.long) for s in starts], dim=0)
        return x.to(dev), y.to(dev)

    def save(self, path: str, step: int = 0, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta: Dict[str, Any] = {
            "step": int(step),
            "vocab_size": int(self.vocab_size),
            "embed_dim": int(self.embed_dim),
            "device": str(self.device),
        }
        if extra_meta:
            meta.update(extra_meta)
        torch.save({"model": self.state_dict(), "meta": meta}, path)

    def load(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        if map_location is None:
            map_location = str(self.device)
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["model"], strict=True)
        meta = ckpt.get("meta", {})
        return meta


class MyTokenizer:
    def __init__(self):
        # char-level 256 vocab
        chars = [chr(i) for i in range(256)]
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)