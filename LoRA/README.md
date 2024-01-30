# LoRA: Low-Rank Adaptation of Large Language Models
- paper review
- [`arXiv`](https://arxiv.org/abs/2106.09685)

### The simple architecture of LoRA
<img src = "https://github.com/AITE-R/paper-review/blob/main/LoRA/figures/figure1.png?raw=true" width=500>

### Summary
- LLM을 한정된 하드웨어 리소스만으로 Fine-tuning할 수 있게 다양한 Adapter method가 제안됨
- 그러나 이들은 다양한 문제점을 안고 있었고 LoRA는 이를 해결
- $W_0\in \mathcal{R}^{d \times k}$는 pre-trained weight, $x$는 각 module의 input이라 한다면 LoRA는 다음과 같이 표현 가능
$$h=W_0 x + \Delta W x = W_0 x + BA x$$
- 이때 $B\in \mathcal{R}^{d\times r}$, $A\in \mathcal{R}^{r\times k}$이며 trainable parameter
- 그리고 r은 hyperparameter이며 LoRA의 rank를 결정짓는 요소
- LoRA는 다양한 장점을 보유
    - 적은 수의 trainable parameter
    - Efficient memory storage usage
    - No Additional Inference Latency
- GPT-2와 GPT-3 등의 LLM을 가지고 이전의 adapter method들과 비교한 결과, LoRA는 압도적인 효율성을 보여줌