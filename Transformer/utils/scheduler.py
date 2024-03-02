from torch.optim.lr_scheduler import _LRScheduler


class TransformerCustomScheduler(_LRScheduler):
    """
    Implement the learning rate scheduler from the paper "Attention is All You Need"
    ```math
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    ```

    Args:
        optimizer (torch.optim): the optimizer to be used
        d_model (int): the dimension of the model
        warmup_epochs (int): the number of warmup epochs (default `3`)
        last_epoch (int): the index of the last epoch (default `-1`)

    Returns:
        The learning rate scheduler explained "Attention is All You Need"

    Examples:
    >>> optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    >>> scheduler = TransformerCustomScheduler(optimizer, d_model=512, warmup_steps=4000)
    """

    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_epochs: int = 3,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = (self.d_model**-0.5) * min(
            step_num**-0.5, step_num * self.warmup_epochs**-1.5
        )
        return [lr for _ in self.base_lrs]


class PolynomialLRDecay(_LRScheduler):

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=1e-5, power=0.9):
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps should be greater than 1.")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.end_learning_rate)
                * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group["lr"] = lr
