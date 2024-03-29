import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int,
        drop_prob: float = 0.1,
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
