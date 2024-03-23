## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- paper review
- [`arXiv`](https://arxiv.org/abs/1810.04805) | [`Github`](https://github.com/google-research/bert)

## Pre-training and Fine-tuning
<img src = "https://github.com/AITE-R/paper-review/blob/main/BERT/figures/figure1.png" width=600>

### Summary
- NLP의 모든 task에서 사용 가능한 Backbone, Foundation인 BERT를 제안
- model의 architecture로는 Transformer의 Encoder를 채택
- 또한 Bidirectional representation을 학습하는 방식을 제안
- 이를 통해 두 문장 사이의 관계를 이해하는 다양한 task에서 뛰어난 성능을 보여줌
- 또한 model size가 커질수록 성능이 비례적으로 증가해 리소스만 충분하다면 좋은 성능 보장 가능