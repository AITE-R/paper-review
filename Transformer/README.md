# Attention Is All You Need
- paper review
- [`arXiv`](https://arxiv.org/abs/1706.03762), [`Github`](https://github.com/tensorflow/tensor2tensor)

### The architecture of Transformer
<img src = "https://github.com/AITE-R/paper-review/assets/91061904/2917e6bb-abed-4850-901a-2e1e15619026" width=300>

### Summary
- Transforemr는 self-attention만을 사용해 representation learning을 하는 첫 번째 모델이다.
- self-attention을 사용함에 따라 RNN의 recurrent connection을 끊고 성능 향상을 이루었다.
- recurrent connection을 끊음에 따라 position information을 주입하기 위해 positional encoding을 수행하고 이를 단어 임베딩 벡터와 summation한다.
- 특정 시점에서 이전 시점의 정보를 활용하는 auto-regressive한 모델이다.
$$p(\mathbf{x}) = \prod\limits_{i=1}^{n}p(x_i \vert x_1, x_2, \ldots, x_{i-1}) =
\prod\limits_{i=1}^{n} p(x_i \vert \mathbf{x}_{< i } )$$
