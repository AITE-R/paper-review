## AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning

- paper review
- [`arXiv`](https://arxiv.org/abs/2303.10512) | [`Github`](https://github.com/QingruZhang/AdaLoRA)

### Summary

- 이 논문을 단 한 문장으로 요약한다면 다음과 같이 쓸 수 있다.

  ***`<p style="text-align: center;">`How can we allocate the parameter budget adaptively according to importance of modules to improve the performance of parameter-efficient fine-tuning?`</p>`***


- 즉, LoRA와 달리 성능을 효율적으로 향상시킬 수 있는 module에 집중하자는 것이다.
- 메인 아이디어인 AdaLoRA를 수식으로 표현하면 다음과 같이 쓸 수 있다.

$$W = W^{(0)} + \Delta = W^{(0)} + P\Lambda Q$$

- 이때 $P \in \mathcal{R}^{d_1 \times r}$, $Q\in \mathcal{R}^{r\times d_2}$이고 $\Lambda\in \mathcal{R}^{r\times r}$는 diagonal matrix

- 그리고 $P^TP = QQ^T = I$를 유지하기 위해 다음의 regularizer를 활용
$$R(P, Q)=\Vert P^TP-I\Vert_{F}^2+\Vert QQ^T-I\Vert_F^2$$

- 두 번째 아이디어인 importance metric에 대해 살펴보자.
- singular value는 다음과 같은 기준으로 pruning을 수행
$$\Lambda_k^{(t+1)}= \Tau \left(\tilde{\Lambda}_k^{(t)}, S_k^{(t)}\right)$$

- 자세한 것은 리뷰 자료를 참고

- 이때 importance score는 다음과 같이 구한다.
$$S_{k,i}=s\left(\lambda_{k,i}\right)+\frac{1}{d_1}\sum_{j=1}^{d_1}s\left(P_{k,ji}\right)+\frac{1}{d_2}\sum_{j=1}^{d_2}s\left(Q_{k, ij}\right)$$

- 그리고 논문에서는 sensitivity를 다음과 같이 구하는 것을 제안한다.

$$\bar{I}^{(t)}\left(w_{ij}\right)=\beta_1 \bar{I}^{(t-1)}\left(w_{ij}\right)+\left(1-\beta_1\right) I^{(t)}\left(w_{ij}\right)$$

$$\bar{U}^{(t)}\left (w_{ij} \right)=\beta_2 \bar{U}^{(t-1)} \left(w_{ij} \right) + \left(1-\beta_2 \right) \vert I^{(t)} \left(w_{ij}\right) - \bar{I}^{(t)} \left(w_{ij} \right) \vert$$

- 이렇게 $\bar{I}^{(t)}$와 $\bar{U}^{(t)}$를 구하면 importance score는 다음과 같이 정의한다.

$$s^{(t)}\left(w_{ij}\right)=\bar{I}^{(t)}\left(w_{ij}\right)\cdot \bar{U}^{(t)}\left(w_{ij}\right)$$
