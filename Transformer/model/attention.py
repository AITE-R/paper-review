import math
from typing import Optional

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        batch_size, head, length, d_tensor = k.size()

        score = q @ k.transpose(2, 3) / (math.sqrt(d_tensor) + eps)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)
        v = score @ v
        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # step 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # step 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # step 3. do scale dot product to compute similiarity
        out, attention = self.attention(q, k, v, mask=mask)

        # step 4. concatenate and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, d_miodel = tensor.size()

        d_tensor = d_miodel // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
