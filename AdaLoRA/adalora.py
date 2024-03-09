import torch
import torch.nn as nn


class AdaLoRA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 0,
        alpha: int = 1,
    ) -> None:
        super(AdaLoRA, self).__init__()

        self.scaling = alpha if alpha > 0 else float(r)

        self.lora_U = nn.Parameter(torch.randn(in_dim, r), requires_grad=True)
        self.lora_E = nn.Parameter(torch.randn(r, 1), requires_grad=True)
        self.lora_V = nn.Parameter(torch.randn(r, out_dim), requires_grad=True)

        self._init_params_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        UEV = torch.matmul(self.lora_U, self.lora_E * self.lora_V) * self.scaling
        UEVx = torch.matmul(UEV, x)
        return UEVx

    def _init_params_(self):
        nn.init.zeros_(self.lora_E)
        nn.init.normal_(self.lora_U, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_V, mean=0.0, std=0.02)


class AdaLoRALinear(nn.Module):
    def __init__(self, layer: nn.Module, r: int, alpha: int, dropout: float) -> None:
        super(AdaLoRALinear, self).__init__()
        out_dim = layer.in_features
        in_dim = layer.out_features

        self.pretrained_weight = nn.Parameter(
            layer.weight.data.clone(), requires_grad=False
        )
        self.adalora = AdaLoRA(in_dim, out_dim, r, alpha)
        self.dropout = nn.Dropout(dropout)

        layer.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = torch.matmul(self.pretrained_weight, x)
        Wx = self.dropout(Wx)
        UEVx = self.adalora(x)
        return Wx + UEVx


class AdaLoRATransformer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        r: int = 8,
        alpha: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super(AdaLoRATransformer, self).__init__()
        self.adalora_layers = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                self.adalora_layers.append(AdaLoRALinear(layer, r, alpha, dropout))
            else:
                self.adalora_layers.append(layer)
        self.adalora_layers = nn.ModuleList(self.lora_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.adalora_layers:
            x = layer(x)
        return x


class AdaLoRARegularizer(nn.Module):
    def __init__(self, model: nn.Module, regular_lambda: float) -> None:
        super(AdaLoRARegularizer, self).__init__()
        self.model = model
        self.regular_lambda = regular_lambda

    def forward(self) -> torch.Tensor:
        regular_loss, num_params = 0, 0
        for name, params in self.model.named_parameters():
            if "lora_U" in name or "Lora_V" in name:
                if "lora_U":
                    M = torch.matmul(params, params.t())
                else:
                    M = torch.matmul(params.t(), params)
                I = torch.eye(*M.size(), device=params.device)
                I.requires_grad = False
                regular_loss += self.regular_lambda * torch.norm(M - I, p="fro")
                num_params += 1
        return regular_loss / num_params


class RankAllocator(object):
    def __init__(self):
        pass

    def schedule_threshold(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def calculate_score(self):
        raise NotImplementedError

    def mask_to_target_rank(self):
        raise NotImplementedError

    def update_and_mask(self):
        raise NotImplementedError
