import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 8,
        alpha: int = 32,
    ) -> None:
        super(LoRA, self).__init__()

        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.randn(in_dim, r), requires_grad=True)
        self.lora_B = nn.Parameter(torch.randn(r, out_dim), requires_grad=True)
        self._reset_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BA = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        BAx = torch.matmul(BA, x)
        return BAx

    def _reset_params(self):
        nn.init.zeros_(self.lora_B)
        nn.init.normal_(self.lora_A)


class LoRALinear(nn.Module):
    def __init__(self, layer: nn.Module, r: int, alpha: int, dropout: float) -> None:
        super(LoRALinear, self).__init__()
        out_dim = layer.in_features
        in_dim = layer.out_features

        self.pretrained_weight = nn.Parameter(
            layer.weight.data.clone(), requires_grad=False
        )
        self.lora = LoRA(in_dim, out_dim, r=r, alpha=alpha)
        self.dropout = nn.Dropout(p=dropout)

        layer.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = torch.matmul(self.pretrained_weight, x)
        Wx = self.dropout(Wx)
        BAx = self.lora(x)
        return Wx + BAx


class LoRATransformer(nn.Module):
    def __init__(self, model: nn.Module, r: int, alpha: int, dropout: float) -> None:
        super(LoRATransformer, self).__init__()
        self.lora_layers = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                lora_layer = LoRALinear(layer, r, alpha, dropout)
                self.lora_layers.append(lora_layer)
        self.lora_layers = nn.ModuleList(self.lora_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.lora_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    transformer = nn.Transformer()
    lora_transformer = LoRATransformer(transformer, r=8, alpha=32, dropout=0.05)
    num_total_params = sum(p.numel() for p in transformer.parameters())
    num_trainable_params = sum(
        p.numel() for p in lora_transformer.parameters() if p.requires_grad
    )
    print(f"# Total Parameters: {num_total_params}")
    print(f"# Trainable Parameters: {num_trainable_params}")
    print(f"# Ratio: {num_trainable_params / num_total_params:.2f}")
    """
    # Total Parameters: 44,140,544 (44.14M)
    # Trainable Parameters: 638,976 (0.64M)
    # Ratio: 0.01
    """
