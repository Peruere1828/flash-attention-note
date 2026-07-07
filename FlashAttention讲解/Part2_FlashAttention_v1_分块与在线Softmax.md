# Part 2: FlashAttention v1 —— 分块与在线 Softmax

> **适合读者**：理解了 Part 1（Attention 的瓶颈是 HBM 读写），有一定的线性代数基础。
> **目标**：完全掌握 FlashAttention v1 的核心——在线 softmax 及其分块实现。

---

## 2.1 问题重述

标准 Attention 的流程中，S = Q @ K^T 的结果要写回 HBM，softmax 时再读出来。如果我们**不分块**，就必须把整个 S(N×N) 存在 HBM 里。

如果分块计算呢？

```
┌─────────┐  ┌─────┬─────┬─────┐
│ Q (Br×d)│  │ K₁  │ K₂  │ K₃  │  ← K 按列分块，每块 Bc 列
└─────────┘  └─────┴─────┴─────┘
      │           │     │     │
      └─────┬─────┘     │     │
            ▼           ▼     ▼
      ┌──────────┬──────────┬──────────┐
      │ S₁(Br×Bc)│ S₂(Br×Bc)│ S₃(Br×Bc)│  ← 分块计算的得分
      └──────────┴──────────┴──────────┘
```

问题来了：**softmax 需要对整行求和分母**。如果只能看到一小块 S₁，怎么能正确计算 softmax？

答案是：**不能一次性算对，但可以通过增量更新逐步修正**。

---

## 2.2 回顾一下数值稳定的 Softmax

标准 softmax 的数值稳定版本：

```python
# 对一个长度为 B 的向量 x:
def stable_softmax(x):
    m = max(x)                    # 最大值（防止上溢）
    f = [exp(xi - m) for xi in x] # 减最大值后再 exp
    l = sum(f)                    # 求和项（全局归一化因子）
    return [fi / l for fi in f]   # 归一化
```

需要两个统计量：
- **m**：向量最大值
- **l**：exp 值的和（即分母）

这两者需要**看到所有元素**才能计算。

---

## 2.3 在线 Softmax 的核心思想

假设我们有一个长度为 2B 的向量 x = [x⁽¹⁾, x⁽²⁾]，先看前半段再后半段。

**步骤 1：处理 x⁽¹⁾**

```
m₁ = max(x⁽¹⁾)                    ← 局部最大值
f₁ = exp(x⁽¹⁾ - m₁)              ← 局部 exp
l₁ = sum(f₁)                      ← 局部求和
softmax_local(x⁽¹⁾) = f₁ / l₁    ← 局部 softmax（还不是最终的！）
```

**步骤 2：处理 x⁽²⁾，同时修正 x⁽¹⁾ 的结果**

计算 x⁽²⁾ 的局部值：

```
m₂ = max(x⁽²⁾)
f₂ = exp(x⁽²⁾ - m₂)
l₂ = sum(f₂)
```

现在有了两个局部最大值 m₁ 和 m₂，**全局最大值**：

```
m_new = max(m₁, m₂)
```

**关键修正**：之前 f₁ 和 l₁ 减的是 m₁，现在应该减 m_new。修正方式是**乘上一个缩放因子**：

```
# 修正 x⁽¹⁾ 的 exp 值（分子）
f₁_corrected = f₁ × exp(m₁ - m_new)   # 把减 m₁ 变成减 m_new

# 修正 x⁽¹⁾ 的求和项（分母）
l₁_corrected = l₁ × exp(m₁ - m_new)

# 修正 x⁽²⁾ 的 exp 值和求和项
f₂_corrected = f₂ × exp(m₂ - m_new)
l₂_corrected = l₂ × exp(m₂ - m_new)

# 全局求和
l_new = l₁_corrected + l₂_corrected

# 最终 softmax
softmax(x⁽¹⁾) = f₁_corrected / l_new
softmax(x⁽²⁾) = f₂_corrected / l_new
```

**核心洞察**：只需要保存两个标量（m 和 l）和已经计算过的 softmax 结果，就可以在下一次迭代时修正之前的结果——**无需回头重新扫描原始数据**。

---

## 2.4 证明一下为什么这样是对的

为什么 f₁_corrected = f₁ × exp(m₁ - m_new) 就是正确的"减全局最大值"版本？

```
原始定义： f₁_corrected[i] = exp(x⁽¹⁾[i] - m_new)

展开：    = exp(x⁽¹⁾[i] - m₁ + m₁ - m_new)    ← 加一项再减一项
         = exp(x⁽¹⁾[i] - m₁) × exp(m₁ - m_new) ← exp(a+b)=exp(a)×exp(b)
         = f₁[i] × exp(m₁ - m_new)            ← 果然就是乘一个修正因子！
```

**魔法就在于 exp 的性质**：exp(a+b) = exp(a) × exp(b)。这让我们可以在事后修正 exp 值。

---

## 2.5 把在线 Softmax 整合进 Attention

现在我们把上面的思路扩展到完整的 Attention 计算中。不仅要算 softmax，还要算 O = P @ V。

直觉上，我们需要 **同时维护 O 的增量更新**。分块计算的核心循环如下（参考原始论文 Algorithm 1）：

```python
import numpy as np


def flash_attention_forward(Q, K, V, M, softmax_scale=None):
    """FlashAttention v1 前向传播算法

    参数:
        Q, K, V: 形状为 (N, d) 的 HBM 矩阵
        M: SRAM 可容纳的元素个数
        softmax_scale: 默认 1/sqrt(d)，Standard Attention 中的缩放因子
    """
    N, d = Q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / np.sqrt(d)

    # 1. 根据 SRAM 大小计算分块的 Block size
    #    约束：Qᵢ(Br×d) + Kⱼ(Bc×d) + Vⱼ(Bc×d) + Oᵢ(Br×d) ≤ M
    #    设 Br = Bc = B，则 4Bd ≤ M → B ≤ M/(4d)
    B = M // (4 * d)
    Br, Bc = B, B

    # 2. 在 HBM 中初始化输出矩阵和统计量
    O = np.zeros((N, d))
    l = np.zeros(N)  # 归一化分母，初始化为 0
    m = np.full(N, -np.inf)  # 局部最大值，初始化为负无穷

    Tc = int(np.ceil(N / Bc))
    Tr = int(np.ceil(N / Br))

    # 4. 外循环：遍历 K, V 的块（列块）
    for j in range(Tc):
        # 5. 从 HBM 加载 K_j, V_j 到 SRAM
        K_j = K[j * Bc : (j + 1) * Bc, :]  # 形状: (Bc, d)
        V_j = V[j * Bc : (j + 1) * Bc, :]  # 形状: (Bc, d)

        # 6. 内循环：遍历 Q, O 的块（行块）
        for i in range(Tr):
            # 7. 从 HBM 加载 Q_i, O_i, l_i, m_i 到 SRAM
            Q_i = Q[i * Br : (i + 1) * Br, :]  # 形状: (Br, d)
            O_i = O[i * Br : (i + 1) * Br, :]  # 形状: (Br, d)
            l_i = l[i * Br : (i + 1) * Br][:, np.newaxis]  # 转为列向量 (Br, 1)
            m_i = m[i * Br : (i + 1) * Br][:, np.newaxis]  # 转为列向量 (Br, 1)

            # 8. 计算当前块的注意力得分（带 scaling）
            S_ij = (Q_i @ K_j.T) * softmax_scale  # 形状: (Br, Bc)

            # 9. 计算当前块的局部统计量
            m_hat = np.max(S_ij, axis=1, keepdims=True)  # 形状: (Br, 1)
            P_hat = np.exp(S_ij - m_hat)  # 形状: (Br, Bc)
            l_hat = np.sum(P_hat, axis=1, keepdims=True)  # 形状: (Br, 1)

            # 10. 计算融合后的新全局最大值和新分母
            m_new = np.maximum(m_i, m_hat)  # 形状: (Br, 1)

            alpha = np.exp(m_i - m_new)
            beta = np.exp(m_hat - m_new)

            l_new = alpha * l_i + beta * l_hat  # 形状: (Br, 1)

            # 11. 修正旧的 O_i 并融合新块的注意力结果，最后统一除以 l_new
            # 解释：(alpha * l_i * O_i) 恢复了未归一化的旧加权和并进行了统一尺度缩放
            #       (beta * P_hat @ V_j) 是当前块新贡献的统一尺度缩放后的加权和
            O_i = (alpha * l_i * O_i + beta * (P_hat @ V_j)) / l_new

            # 12. 更新统计量
            m_i = m_new
            l_i = l_new

            # 13. 将更新后的数据写回 HBM
            O[i * Br : (i + 1) * Br, :] = O_i
            l[i * Br : (i + 1) * Br] = l_i.squeeze()
            m[i * Br : (i + 1) * Br] = m_i.squeeze()

    return O
```

**关键实现细节**：

- **alpha / beta 缩放因子**：`alpha = exp(mᵢ - m_new)` 和 `beta = exp(m̂ⱼ - m_new)` 将旧值和新值统一校准到新的全局最大值 m_new 下。由于 exp(a)/exp(b) = exp(a-b)，这比直接除更数值稳定。
- **列向量广播**：`l_i` 和 `m_i` 通过 `[:, np.newaxis]` 转为 (Br, 1) 列向量，与 O_i (Br, d) 运算时自动逐行广播，每行独立维护自己的归一化因子。
- **softmax_scale**：`1/√d` 是 Standard Attention 的标准缩放，防止点积过大导致 softmax 梯度消失。

---

## 2.6 这个算法节省了多少 HBM 读写？

标准的 Attention：

| 操作     | HBM 访问量                    |
| -------- | ----------------------------- |
| Q @ K^T  | 读 Q,K: 2Nd → 写 S: N²        |
| softmax  | 读 S: N² → 写 P: N²           |
| P @ V    | 读 P: N², 读 V: Nd → 写 O: Nd |
| **总计** | **4Nd + 4N²**                 |

FlashAttention v1（假设 Br = Bc = M/4d）：

| 操作                    | HBM 访问量               |
| ----------------------- | ------------------------ |
| Q 只从 HBM 读一次       | Nd                       |
| K, V 各读 Tc = 4Nd/M 次 | (Nd) × (4Nd/M) = 4N²d²/M |
| O, l, m 的读写          | ≈ Nd                     |
| **总计**                | **O(N²d²/M)**            |

> 因为 SRAM 大小 M（~100KB）远大于 d（~64-128B 行大小），所以 N²d²/M < N²，节省大量 HBM 访问。

具体数值比较（N=4096, d=64, M=100KB）：

| 方法              | HBM 访问量         | 相对加速 |
| ----------------- | ------------------ | -------- |
| Standard          | ~4Nd + 4N² ≈ 68M   | 1x       |
| FlashAttention v1 | 约 ½~¼ 的 HBM 访问 | 2-4x     |

---

## 2.7 从论文代码到实际源码

现在来看看 FA4 仓库中的实际代码。虽然 FA4 的目标架构更先进（Hopper/Blackwell），但 v1 的核心分块思路依然贯穿其中。

看 [flash_attn/cute/softmax.py](../flash_attn/cute/softmax.py) 中的在线 softmax 实现：

```python
# 简化的在线 softmax 核心逻辑
class Softmax:
    def __init__(self, ...):
        self.row_max = ...   # m: 每行的最大值
        self.row_sum = ...   # l: 每行的 exp 求和项
        
    def step(self, scores_chunk, ...):
        # 处理一个分块的 attention score
        m_local = scores_chunk.max(dim=-1)           # 当前块的行最大值
        p_local = exp(scores_chunk - m_local)        # 局部 exp
        l_local = p_local.sum(dim=-1)                # 局部求和
        
        m_new = max(self.row_max, m_local)           # 更新全局最大值
        scale_old = exp(self.row_max - m_new)        # 旧值修正因子
        scale_new = exp(m_local - m_new)             # 新值修正因子
        
        self.row_sum = self.row_sum * scale_old + l_local * scale_new
        self.row_max = m_new
        
        # 更新输出 O（注意：此处 output 存的是未归一化的分子）
        # FA4 采用 v2 式的"惰性归一化"：step() 只累加分子，normalize() 最后统一除分母
        output = output * scale_old + (p_local * scale_new) @ V_chunk
```

> **注意区分**：FA4 的 `Softmax.step()` 中的 `output` 存的是**未归一化的分子**（v2 惰性归一化），而 2.5 节 v1 伪代码中的 `Oᵢ` 存的是**每一步都归一化**的结果。这正是 2.8 节讨论的"v1 多余的除法"问题。

在 FA4 的实际 SM80（Ampere）实现中，这个分块循环体现在 [flash_attn/cute/flash_fwd.py](../flash_attn/cute/flash_fwd.py)：

```python
class FlashAttentionForwardSm80:
    def forward(self, q, k, v, ...):
        # ... 
        for j in range(num_n_blocks):   # 外循环遍历 KV 块
            k_block, v_block = load_kv_block(k, v, j)
            
            for i in range(num_m_blocks):  # 内循环遍历 Q 块
                q_block = load_q_block(q, i)
                
                # 计算当前块的 attention score
                scores = q_block @ k_block.T
                
                # 在线 softmax 更新
                self.softmax.step(scores, v_block)  
                
                # 累加输出
                output[i] = self.softmax.normalize(output[i])
```

> **注意**：FA4 中 SM90 和 SM100 的实现使用了更高级的 TMA（Tensor Memory Accelerator）来搬运数据，而不是手工的 cpasync。我们将在 Part 4 中深入。

---

## 2.8 v1 的局限

虽然 v1 已经是重大突破，但它有几个问题：

### 1. 前向过程中多余的除法

在每一步更新 O 时，都除以了当前的 l_new。但如果后面还有更多 KV 块要处理，这个除法是多余的——下次更新时又要乘以旧的分母再除以新的。**v2 把这个除法推迟到了最后一步**（惰性归一化）。

### 2. 内外循环顺序导致 Q 反复加载

外循环 j 遍历 KV 块，内循环 i 遍历 Q 块。这意味着每个外循环步都要重新加载所有 Q 块（因为上一步的 Q 块计算完后被丢弃了）：
- Q 被加载了 Tc = N/Bc 次
- 如果 N 很大，Tc 也很大，Q 的反复加载成为瓶颈

**v2 交换了内外循环的顺序**，让 Q 只加载一次。

### 3. Warp 级别的工作分配

在一个 thread block 内的 4 个 warp 之间，v1 是分割 K 和 V 的列，各算各的结果再求和——需要 warp 间通信。**v2 改为分割 Q 的行**，各算各的结果直接拼起来，无需通信。

---

## 2.9 关键总结

1. **在线 softmax** 是 FlashAttention 的数学基石——通过保存 m（最大值）和 l（求和项）两个标量，可以在分块情况下逐步修正 softmax 结果
2. **双重循环**：外循环遍历 KV 块，内循环遍历 Q 块，确保每次 SRAM 加载的数据被充分复用
3. **精确性**：FlashAttention 是精确算法，不改变计算结果
4. v1 的局限：多余的除法、Q 反复加载、warp 间通信开销 —— 这三个问题分别在算法、并行度和 warp 分片上被 v2 解决

---

> **下一讲**：Part 3 - FlashAttention v2 的三个关键改进（算法优化、循环交换、warp 分片）
