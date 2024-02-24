import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int,
        tgt_pad_idx: int,
        tgt_sos_idx: int,
        enc_voc_size: int,
        dec_voc_size: int,
        d_model: int,
        n_head: int,
        max_len: int,
        ffn_hidden: int,
        n_layers: int,
        drop_prob: float,
        device: str,
    ) -> None:
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.device = device

        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device,
        )

        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device,
        )

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.size(1)
        tgt_sub_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len))
            .type(torch.ByteTensor)
            .to(self.device)
        )
        return tgt_pad_mask & tgt_sub_mask
