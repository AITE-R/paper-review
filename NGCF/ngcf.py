import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super(EmbeddingLayer, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self._init_weight()

    def _init_weight(self) -> None:
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self) -> torch.Tensor:
        return torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )


class EmbeddingPropagationLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        message_dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super(EmbeddingPropagationLayer, self).__init__()
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(message_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self._init_weight()

    def _init_weight(self) -> None:
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.zeros_(self.W2.bias)

    def forward(self, embeddings: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # side = (L + I) E
        side = torch.sparse.mm(norm_adj, embeddings)

        sum_emb = self.W1(side)  # transformation message
        bi_emb = self.W2(embeddings * side)  # bi-interaction message

        out = self.leaky_relu(sum_emb + bi_emb)
        out = self.dropout(out)
        out = F.normalize(out, p=2, dim=1)
        return out


class BPRLoss(nn.Module):
    def __init__(self, reg: float = 1e-5) -> None:
        super(BPRLoss, self).__init__()
        self.reg = reg

    def forward(
        self, user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor
    ) -> torch.Tensor:
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = self.reg * (
            user_emb.norm(p=2).pow(2)
            + pos_emb.norm(p=2).pow(2)
            + neg_emb.norm(p=2).pow(2)
        )
        return mf_loss + reg_loss


class NGCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        layer_dims: tuple = (64, 64, 64),
        node_dropout: float = 0.1,
        message_dropout: float = 0.1,
        negative_slope: float = 0.2,
        reg: float = 1e-5,
    ) -> None:
        super(NGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.node_dropout = node_dropout

        # 1. embedding layer
        self.embedding = EmbeddingLayer(num_users, num_items, embedding_dim)

        # 2. embedding propagation layers
        self.propagation_layers = nn.ModuleList()
        dims = [embedding_dim] + list(layer_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.propagation_layers.append(
                EmbeddingPropagationLayer(
                    in_dim,
                    out_dim,
                    message_dropout,
                    negative_slope,
                )
            )

    def forward(self, norm_adj: torch.Tensor) -> tuple:
        if self.training and self.node_dropout > 0:
            norm_adj = self._sparse_dropout(norm_adj, self.node_dropout)

        ego = self.embedding()
        all_embeddings = [ego]

        for layer in self.propagation_layers:
            ego = layer(ego, norm_adj)
            all_embeddings.append(ego)

        all_embeddings = torch.cat(all_embeddings, dim=1)
        user_all, item_all = torch.split(
            all_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_all, item_all

    def _sparse_dropout(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        x = x.coalesce()
        mask = torch.rand(x._nnz(), device=x.device) >= rate
        indices = x.indices()[:, mask]
        values = x.values()[mask] / (1.0 - rate)
        return torch.sparse_coo_tensor(indices, values, x.shape)
