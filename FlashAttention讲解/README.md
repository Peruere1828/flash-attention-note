# FlashAttention 深入讲解

从 GPT 基础到 FA4 Blackwell 实现的完整学习路径。每篇都直指仓库源代码。

---

## 目录

| # | 文件 | 内容 | 适合读者 |
|---|------|------|---------|
| 1 | [Part1_从GPT的Attention说起](Part1_从GPT的Attention说起.md) | Attention 为什么慢？HBM vs SRAM，MAC 瓶颈 | GPT 基础 |
| 2 | [Part2_FlashAttention_v1_分块与在线Softmax](Part2_FlashAttention_v1_分块与在线Softmax.md) | 在线 softmax 推导、分块算法、MAC 分析 | 基础线性代数 |
| 3 | [Part3_FlashAttention_v2_三项改进](Part3_FlashAttention_v2_三项改进.md) | 惰性归一化、循环交换、Q 分片 | 理解 v1 |
| 4 | [Part4_走进FA4代码](Part4_走进FA4代码.md) | interface.py 调度、TMA、2CTA、TCGEN05 | Python/CUDA 基础 |
| 5 | [Part5_前沿进展_MLA与FP8](Part5_前沿进展_MLA与FP8.md) | DeepSeek MLA、FP8 推理、未来方向 | 关注业界进展 |

## 阅读建议

- **想快速理解核心思路**：Part 1 + Part 2
- **想理解 v1→v2 的改进动机**：Part 3
- **想读源码**：先 Part 2，再 Part 4
- **对 DeepSeek 感兴趣**：Part 5

## 参考材料

- [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention v2 论文](https://arxiv.org/abs/2307.08691)
- FA4 仓库：[flash_attn/cute/](../flash_attn/cute/)
