import torch

## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

## model parameter
batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

## optimizer parameter
init_lr = 1e-5
factor = 9e-1
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float("inf")
