from conf import *
from utils.dataloader import DataLoader
from utils.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(
    ext=(".en", ".de"),
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
    init_token="<sos>",
    eos_token="<eos>",
    path="/home/jeongseok/aite-r/paper-review/Transformer/data/wmt2016"
)

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test, batch_size=batch_size, device=device
)

src_pad_idx = loader.source.vocab.stoi["<pad>"]
trg_pad_idx = loader.target.vocab.stoi["<pad>"]
trg_sos_idx = loader.target.vocab.stoi["<sos>"]

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
