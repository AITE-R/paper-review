# PageRank: Standing on the Shoulders of Giants
- paper review
- [`arXiv`](https://dl.acm.org/doi/10.1145/1953122.1953146)

### Summary
- PageRank는 웹 페이지 간에 하이퍼링크로 연결되어있다는 특징을 활용해 웹페이지의 중요도를 측정하는 알고리즘이다.
- PageRank 알고리즘 과정에 있어 해의 존재성, 유일성, 효율적인 계산 세가지 검증을 수행했다.
- 'Standing on the Shoulders of Giants'의 의미는 선행된 연구들의 지식을 기반으로 페이지랭크가 성공할 수 있었음을 내포한다.
- HITS도 마찬가지로 연결정보를 활용한 웹 페이지 랭킹 알고리즘이다. Hub, authority score 두가지 measure로 축정한다.

<div align="center">
    <img width="180" alt="Untitled" src="https://github.com/AITE-R/paper-review/assets/91061904/fb1b3ac7-5684-4fed-ba7d-e5b4a03d0290">
</div>

$$ \begin{bmatrix}
\pi_a & \pi_b & \pi_c & \pi_d
\end{bmatrix}= \begin{bmatrix}
\pi_a & \pi_b & \pi_c & \pi_d
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 0 & 0  \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\ 
\frac{1}{2} & \frac{1}{2} & 0 & 0 \\ 
\end{bmatrix}$$
