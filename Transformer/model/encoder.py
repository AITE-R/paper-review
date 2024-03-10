import torch
import torch.nn as nn

from .ffn import PositionwiseFeedForward
from .attention import MultiHeadAttention
from .embedding import TransformerEmbedding


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        drop_prob: float,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        n_layers: int,
        drop_prob: float,
        device: str,
    ) -> None:
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(
            vocab_size=enc_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
