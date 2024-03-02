from typing import Tuple, List

from torchtext.data import Field, BucketIterator
from torchtext.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(
        self,
        path: str,
        ext: Tuple[str],
        tokenize_en: List[str],
        tokenize_de: List[str],
        init_token: str = "<sos>",
        eos_token: str = "<eos>",
    ) -> None:
        assert ext in ((".de", ".en"), ("en", ".de"))
        self.path = path
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token

    def make_dataset(self) -> Multi30k:
        if self.ext == (".de", ".en"):
            tokenize = self.tokenize_de
        else:
            tokenize = self.tokenize_en

        self.source = Field(
            tokenize=tokenize,
            init_token=self.init_token,
            eos_token=self.eos_token,
            lower=True,
            batch_first=True,
        )
        self.target = Field(
            tokenize=tokenize,
            init_token=self.init_token,
            eos_token=self.eos_token,
            lower=True,
            batch_first=True,
        )

        train_data, valid_data, test_data = Multi30k.splits(
            exts=self.ext, fields=(self.source, self.target), path=self.path
        )
        return train_data, valid_data, test_data

    def build_vocab(self, train_data: Multi30k, min_freq: int) -> None:
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_loader(
        self,
        train_data: Multi30k,
        valid_data: Multi30k,
        test_data: Multi30k,
        batch_size: int,
        device: str,
    ) -> Tuple[Multi30k]:
        train_loader, valid_loader, test_loader = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            device=device,
        )
        return train_loader, valid_loader, test_loader
