# 20. 2022-2024：长上下文与高效 Attention

2022年至2024年间，大语言模型的上下文窗口从2K token扩展到128K乃至1M token。这一扩展并非简单的参数调整，而是对注意力机制底层计算效率的系统性重构。标准Attention的二次复杂度是核心障碍——序列长度翻64倍，计算量增长4096倍。FlashAttention系列将优化目标从算法复杂度转向IO效率，Ring Attention将序列切分到多卡并行计算，两者共同推动了长上下文训练从不可能变为工程常规。

## 20.1 标准 Attention 的二次复杂度问题

### 计算与显存的双重瓶颈

Self-Attention的数学形式定义了它的复杂度上限。给定输入序列的Q、K、V矩阵（形状均为N×d，N为序列长度，d为head维度），Attention计算为：

Softmax(QK^T/√d)V

其中QK^T产生一个N×N的注意力矩阵。这个矩阵的**计算复杂度为O(N²d)**，**显存占用为O(N²)**。[^202^]

当序列长度从2K扩展到128K（64倍），计算量增长64²=4096倍。在A100 GPU上，标准Attention处理8K序列已接近显存上限；超过16K则几乎必然溢出。GPT-4初版的32K上下文、Claude 2的100K上下文，在当时都是工程上的巨大挑战。这一增长并非线性外推可以解决的问题——它要求根本性的算法重构。

### 被误解的瓶颈：计算还是IO？

标准Attention的一个反直觉事实是：它并非计算受限（compute-bound），而是**内存带宽受限**（memory-bound）。GPU的HBM（高带宽内存）读写速度远慢于片上SRAM和Tensor Core的计算速度。标准Attention将QK^T的N×N中间结果反复写入HBM再读出，HBM IO复杂度达到O(Nd + N²)。[^203^]

这意味着即使算法层面的FLOP数固定，HBM的往返读写已成为实际瓶颈。FlashAttention的核心洞察正是从这里出发。

## 20.2 FlashAttention：从算法复杂度转向 IO 效率

### FlashAttention v1（2022）：Tiling + Online Softmax

Dao等人提出的FlashAttention并非近似方法——它计算数学上完全等价的注意力输出，只是通过重新组织计算来减少HBM访问。[^203^] 五大关键技术构成了这一突破：

**算子融合**（Operator Fusion）：将整个注意力计算（matmul → softmax → matmul）封装在单个CUDA kernel内，中间结果不写回HBM。

**分块/Tiling**：将Q、K、V划分为适合GPU SRAM（片上缓存，典型大小约100KB-200KB）的块，逐块处理。SRAM带宽比HBM高一个数量级。

**Online Softmax**：通过维护两个running statistics（当前最大值m和指数加权和l），在仅看到部分K块的情况下即可精确累积softmax结果。公式为：

m_new = max(m_old, m_local)
l_new = exp(m_old − m_new) × l_old + exp(m_local − m_new) × l_local

增量式更新保证最终结果与标准softmax完全一致，但无需一次性加载整行。[^204^]

**重计算**（Recomputation）：不存储O(N²)的P矩阵。反向传播时利用保存的softmax统计量（m和l）重新计算所需的中间值。以少量额外计算换取显存的大幅节省。

**HBM IO复杂度分析**：FlashAttention将HBM访问从O(N²)降至O(N² × d² / M)，其中M为SRAM容量。对于典型值d=64、M=100KB、N=4096，HBM访问量减少约9倍。论文证明这是理论下界——FlashAttention在IO复杂度上是最优的。[^204^]

实际加速比：GPT-2 Small（序列1K）1.7×，GPT-2 Large（序列2K）2.1×，4K序列上的OOM问题被彻底解决。[^204^]

### FlashAttention-2（2023）：更好的并行度

FlashAttention-2在v1基础上做了三个关键改进：减少non-matmul FLOP（优化online softmax的除法开销）；在sequence长度维度上增加warp级并行（v1主要在batch/head维度并行）；更均匀的工作分配避免部分warps空闲。[^204^]

在A100上，FlashAttention-2达到理论峰值FLOPS的50-73%，比v1快约1.5-2倍。这一提升并非来自算法复杂度的改变，而是对GPU执行模型的更精细适配。

### FlashAttention-3（2024）：异步流水线与FP8

FlashAttention-3针对H100（Hopper架构）引入了**GEMM-Softmax异步流水线**：构建2-stage pipeline，在计算block j的softmax同时启动block j+1的QK^T GEMM。Softmax使用普通CUDA core，GEMM使用Tensor Core，两类计算资源并行利用。[^204^]

FP8支持是另一关键特性。FlashAttention-3实现了Block Quantization（每块独立计算scale factor）和Incoherent Processing（随机正交矩阵乘以Q/K使值分布均匀化，减少量化误差）。在H100 FP16模式下达到740 TFLOPS/s，FP8模式下达到1.2 PFLOPS/s，理论利用率约75-76%。[^204^]

| 维度 | FlashAttention v1 (2022) | FlashAttention v2 (2023) | FlashAttention v3 (2024) |
|------|-------------------------|-------------------------|-------------------------|
| 核心创新 | Tiling + Online Softmax | Warp级seq并行 + 均匀工作分配 | 异步流水线 + FP8支持 |
| 目标架构 | A100通用 | A100优化 | H100 Hopper专用 |
| HBM IO | 理论最优 O(N²d²/M) | 同v1 | 同v1 |
| A100利用率 | 30-50%峰值FLOPS | 50-73%峰值FLOPS | — |
| H100 FP16性能 | — | — | 740 TFLOPS/s |
| H100 FP8性能 | 不支持 | 不支持 | 1.2 PFLOPS/s |
| 相对加速（vs标准Attention） | 1.7-2.1× | 3-4× | 6-8×（FP8） |
| 关键局限 | 并行度不足 | 无FP8 | Hopper专用 |

FlashAttention三代的演进揭示了一条清晰的优化路径：从算法层面的IO复杂度最小化（v1），到GPU执行模型的精细适配（v2），再到专用硬件特性的充分利用（v3）。每一代都在前一代的基础上推进，但优化目标始终围绕一个核心——减少HBM访问，而非降低算法复杂度。FlashAttention从未改变注意力的数学形式，输出与标准Attention逐位一致，可直接替换任何模型中的注意力实现。

### 复杂度对比图

不同注意力实现的复杂度可以从三个维度比较：算法计算复杂度（FLOP）、HBM IO复杂度（数据搬运量）、显存峰值占用。

```
标准 Attention
  FLOP:        O(N²d) ───────────────────────────── 高
  HBM IO:      O(N² + Nd) ───────────────────────── 极高（反复读写N×N矩阵）
  显存峰值:     O(N² + Nd) ───────────────────────── OOM at N>16K
       │
       ▼
FlashAttention v1/v2/v3
  FLOP:        O(N²d) ───────────────────────────── 相同（精确计算）
  HBM IO:      O(N²d²/M) ───────────────────────── 大幅降低（~M/d²倍）
  显存峰值:     O(Nd) ────────────────────────────── 从O(N²)降至线性
       │
       ▼
Ring Attention + Context Parallelism
  FLOP:        O(N²d) ───────────────────────────── 相同（每卡处理N/cp块）
  HBM IO:      O(N²d²/(M×cp)) ──────────────────── 随cp degree线性降低
  显存峰值:     O(Nd/cp) ─────────────────────────── 每卡显存随cp degree线性降低
  通信开销:     O(N²d/cp) P2P ───────────────────── 环状传递K/V块
       │
       ▼
理论下界（FlashAttention已证明达到）
  HBM IO:      Ω(N²d²/M) ────────────────────────── 不可再降低
```

上图展示了注意力机制优化的清晰层次。标准Attention的HBM IO是平方级，FlashAttention通过分块将其降至与SRAM容量相关的更优界，Ring Attention再通过多卡并行进一步分摊。算法本身的FLOP复杂度始终是O(N²d)，但实际的训练瓶颈已从FLOP转移到了数据搬运。

## 20.3 Ring Attention 与序列并行

FlashAttention解决了单卡上的Attention效率问题，但当序列长度超过单卡显存容量时（即使FlashAttention已将显存从O(N²)降至O(Nd)），必须将序列切分到多卡。这是序列并行（Sequence Parallelism, SP）和上下文并行（Context Parallelism, CP）的使命。

### 序列并行 vs 上下文并行：概念辨析

这两个术语常被混用，实则指向不同层级的切分。

**序列并行（Megatron-LM SP）**：沿序列维度切分LayerNorm、Dropout等非矩阵乘法操作的激活值。与TP共用通信组，目的是减少激活值内存。不处理Attention计算本身。[^260^]

**上下文并行（CP）**：将整个序列切分到多卡进行Attention计算。Ring Attention是CP的一种实现。[^260^][^262^]

两者是正交维度：SP处理"TP层之外"的非计算密集操作，CP处理"Attention层之内"的序列切分。

### DeepSpeed Ulysses（2023）：All-to-All 路线

Ulysses的核心思想：对Q/K/V做all-to-all通信，将序列维度的切分转换为注意力头维度的切分，然后各卡独立计算部分注意力头，最后all-to-all还原。[^252^][^262^]

Ulysses的优势在于通信模式简单（两次all-to-all），且可直接复用现有的多头注意力实现。但它的并行度受注意力头数限制——CP degree不能超过num_heads。对于32头注意力层，最多32路并行。当序列长度达到512K或1M时，这个上限成为瓶颈。[^262^]

### Ring Attention（2023）：环状通信路线

Ring Attention将GPU组织为环形拓扑，每个GPU持有序列的一个chunk。[^258^][^260^] 计算流程如下：

每轮迭代中，每个GPU将自己的K/V块沿环传递给下一个GPU，同时用本地的Q块与接收到的K/V块计算局部注意力。使用online softmax算法累积全局结果。经过cp轮迭代后，每个Q块都与所有K/V块交互完毕。

Ring Attention的关键优势在于：通信与计算重叠——传递K/V的同时进行注意力计算；可扩展到任意序列长度——增加GPU即可；不受注意力头数限制。[^258^][^261^]

实际挑战包括：P2P逐跳通信无法充分利用HPC网络的全带宽；因果掩码导致不同GPU工作量不均（Zig-Zag交错分配可缓解）。[^260^][^261^]

### USP：统一序列并行（2024）

USP（Unified Sequence Parallelism）将Ulysses和Ring Attention统一为一个框架，根据序列长度自动选择策略：短序列用Ulysses（all-to-all通信更少），长序列用Ring Attention（不受head数限制），两者还可以组合使用。实验表明USP可以与ZeRO-3高度兼容，增加SP degree还能降低全局batch size要求，避免大batch对收敛的不利影响。[^252^]

## 20.4 长上下文训练对数据、显存和通信的影响

### 显存占用的重新分布

FlashAttention和序列并行改变了训练中的显存瓶颈结构。标准Attention时代，N×N注意力矩阵是显存大户；FlashAttention将注意力显存降至O(Nd)后，激活值（尤其是FFN层的中间结果）成为新的瓶颈。[^261^]

| 注意力方法 | 注意力显存 | 总激活值显存 | 可支持最大序列长度（单卡80GB） | 多卡扩展方式 |
|-----------|-----------|-------------|------------------------------|-------------|
| 标准Attention | O(N²) | O(N² + N×h) | ~8K（A100已OOM） | 无有效方案 |
| FlashAttention | O(Nd) | O(N×h) | ~64K | 上下文并行(CP) |
| FlashAttention + CP | O(Nd/cp) | O(N×h/cp) | ~1M（cp=128） | 增加cp degree |

h为模型隐藏层维度。表中数据基于典型配置（d=64, h=4096, BF16精度）估算。[^204^][^261^]

从标准Attention到FlashAttention，注意力本身的显存从O(N²)降至O(Nd)——对128K序列，这是从16GB到0.5MB的跃迁。但FFN层的激活值随序列长度线性增长（O(N×h)），在超长序列下仍构成主要压力。上下文并行将激活值均匀分摊到各卡，使得每卡显存随cp degree线性降低，这是百万token训练的基础。

### 长文档数据的构造

长上下文训练不仅需要工程支持，还需要适配的数据。原始网页文本平均长度仅数百token，直接拼接会导致语义断裂。有效的长文档数据构造策略包括：按文档主题聚类后拼接（保持语义连贯性）；在文档边界插入特殊分隔符（让模型学习边界信号）；控制拼接后的序列长度分布（避免全部填满至最大长度）。[^261^]

长序列训练的另一个数据挑战是位置编码的扩展。RoPE、ALiBi等相对位置编码方案允许模型在比训练时更长的序列上推理（外推），但性能会衰减。更可靠的方案是分阶段扩展——先在4K序列上预训练，再在32K上继续训练，最后扩展到128K或更长。每阶段的序列长度增长4-8倍，让模型逐步适应长距离依赖。[^261^]

### 通信开销的量化

序列并行引入的通信不可忽视。以Ring Attention为例，每轮迭代需要传递K/V块，总通信量为O(N²d/cp)。当cp=128、N=1M时，每步的P2P通信量可达数GB。RingX（优化版Ring Attention）在Frontier超算上的测试数据显示：4096 GPU训练1M序列长度，MFU仍可达38%，证明通信与计算重叠的有效性。[^261^]

### 从32K到1M的训练扩展路径

RingX（优化版Ring Attention）在Frontier超算上的实测数据揭示了长上下文训练的扩展效率[^261^]：128K序列使用16 GPU、MFU 44%；256K序列使用32 GPU；512K序列使用64 GPU；1M序列使用128 GPU、MFU 38%。从128K到1M，GPU数增长8倍，MFU仅从44%降至38%。这一小幅下降说明上下文并行的扩展性接近线性。38%的MFU在HPC系统上是长上下文训练的高效率记录。关键优化点包括：使用高带宽网络（InfiniBand或Cray Slingshot）减少P2P延迟；GPU计算与通信的重叠隐藏传输开销；负载均衡策略（Zig-Zag分配）减少空闲等待。

## 20.5 长上下文时代的训练系统变化

### FlashAttention成为事实标准

到2024年，FlashAttention已成为所有主要训练和推理框架的默认注意力实现。PyTorch 2.0内置了scaled_dot_product_attention（SDPA），在底层自动调用FlashAttention。Hugging Face Transformers、Megatron-LM、DeepSpeed均已默认启用FlashAttention变体。从论文发表到成为工业标准仅用不到两年，这一普及速度在大模型基础设施领域极少见。[^204^]

### 从短序列优先到长序列优先的设计转变

长上下文能力已从"加分项"变为"必备项"。GPT-4（128K）、Claude 3（200K）、Gemini 1.5（1M-2M）、DeepSeek-V3（128K）的上下文长度竞赛推动了训练系统的全面重构。

系统层面的关键变化包括：流水线并行从1F1B演进为DualPipe和Zero Bubble Pipeline，目标都是最大化计算-通信重叠，因为长序列下通信占比显著增加。[^326^][^304^] 上下文并行（CP）与张量并行（TP）、流水线并行（PP）、数据并行（DP）、专家并行（EP）组成五维并行网格，成为大规模训练的标准配置。[^281^] 内存优化从"省参数"转向"省激活值"——FlashAttention解决了注意力矩阵的显存问题后，FFN层激活值、KV Cache成为新的优化焦点。

### 长上下文与MoE的交汇

长上下文和MoE（Mixture of Experts）是2024年两大并行演进线，两者在训练系统中交汇时产生新的工程挑战。MoE引入的all-to-all通信（Expert Parallelism）与上下文并行的P2P通信叠加，网络带宽成为硬约束。DeepSeek-V3的解决方案是不使用TP（通过精细内存优化避免了张量并行需求），将PP=16和EP=64作为主要并行维度，在长序列场景下减少了TP与CP的冲突。[^326^]

Megatron-Core 2024+版本提出5D并行（TP+PP+DP+EP+CP），并引入Parallel Folding机制解耦Attention层和MoE层的并行配置，打破了EP≤DP的限制。[^281^][^282^] 这标志着训练系统的并行维度管理已进入精细化编排阶段。

### 推理侧的连锁反应

长上下文训练的成熟直接推动了推理侧KV Cache管理的革新。标准KV Cache随序列长度线性增长，128K上下文下可达数十GB。PagedAttention（vLLM, 2023）将KV Cache分页管理，大幅提升GPU内存利用率。2024年的KV Cache量化、跨层共享、驱逐策略等技术，都是在长上下文成为常态后的必然产物。训练与推理在长上下文场景下的技术协同，成为大模型工程化的新特征。
