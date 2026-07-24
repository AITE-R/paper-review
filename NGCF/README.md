# Neural Graph Collaborative Filtering
- paper review
- [`arXiv`](https://arxiv.org/pdf/1905.08108), [`Github`](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)

### Architecture
<img src = "[https://github.com/AITE-R/paper-review/blob/main/NGCF/figures/figure2.png?raw=true](https://github.com/AITE-R/paper-review/blob/main/NGCF/figures/figure2.png?raw=true)" width=350">

- **Embedding Layer**: User 및 Item feature vector를 가지고 trainable parameter로 초기화
- **Embedding Propagation Layer**: graph 구조를 따라 노드의 정보를 반복적으로 수집하며 노드 간 interaction을 반영해 Emebedding을 정교하게 업데이트
- **high-order connectivity**: 여러 layer를 쌓아 고차원 정보를 학습해 representation을 풍부하게 만듦
- **Prediction**: 각 layer에서 출력된 서로 다른 정보를 concatenation하여 최종 벡터를 구성하고 내적을 통해 선호도 계산
- **Optimization**: BPR Loss를 활용해 user의 선호도 순위를 최적화 및 Over-fitting 방지를 위해 Dropout 및 regularization을 채택
