from typing import Optional

import torch
import torch.nn as nn

from ffn import PositionwiseFeedForward
from attention import MultiHeadAttention
from embedding import TransformerEmbedding


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        drop_prob: int,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: Optional[torch.Tensor],
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # self attention
        _x = x
        x = self.self_attention(q=x, k=x, v=x, mask=tgt_mask)

        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc_out is not None:
            # encoder-decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc_out, v=enc_out, mask=src_mask)

            # add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        n_layers: int,
        drop_prob: float,
        device: str,
    ) -> None:
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(
            vocab_size=dec_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.emb(tgt)

        for layer in self.layers:
            tgt = layer(tgt, enc_src, src_mask, tgt_mask)

        output = self.fc(tgt)
        return output
