# Tokenizer技术演进深度研究报告

> **研究范围**：大模型预训练技术演进书籍第6-8章素材  
> **覆盖章节**：第6章 Tokenizer的核心问题（压缩与泛化）；第7章 BPE、WordPiece、SentencePiece到Byte-level Tokenizer；第8章 2023-2026 Tokenizer的新问题  
> **搜索次数**：12次独立搜索  
> **生成日期**：2025年

---

## 一、关键发现（带引用）

### 1. BPE算法的核心原理与演变

**核心发现**：
- **BPE（Byte Pair Encoding）**最初由Gage于1994年提出作为数据压缩算法，后由Sennrich等人于2016年引入NLP领域用于神经机器翻译[^1^]。它从一个字符级别的初始词汇表开始，迭代地合并训练语料中最频繁的相邻token对，直到达到目标词汇表大小[^2^]。
- BPE目前仍是大多数SOTA LLM的首选分词方法，包括GPT系列、LLaMA、DeepSeek等，原因是其内存需求远低于Unigram方法[^3^]。
- BPE的训练过程是确定性的——给定固定的合并列表，编码是确定性的。这种贪心启发式策略在理论上有局限，但工程上非常高效[^4^]。

**关键文献**：
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units," ACL 2016[^1^]
- Gage, "A New Algorithm for Data Compression," The C Users Journal, 1994[^5^]

### 2. WordPiece与BPE的技术差异

**核心发现**：
- **WordPiece**由Schuster和Nakajima于2012年开发，Google在其神经机器翻译系统和BERT中广泛采用[^6^]。
- **关键区别**：WordPiece与BPE的最大差异在于合并标准。BPE选择频率最高的相邻对进行合并，而WordPiece使用**逐点互信息（Pointwise Mutual Information, PMI）**作为合并标准——优先选择那些共现频率高于各自独立频率预期值的token对[^7^]。
- WordPiece的目标是最大化训练数据的似然，而非简单地最大化频率。这使得算法对语言的语义更敏感，可能产生更有意义的子词单元[^8^]。
- 在BERT的初步实验中，WordPiece相比BPE有微小的性能优势[^9^]。

**关键文献**：
- Schuster & Nakajima, "Japanese and Korean Voice Search," 2012[^6^]
- Wu et al., "Google's Neural Machine Translation System," 2016[^10^]

### 3. SentencePiece的语言无关设计

**核心发现**：
- **SentencePiece**由Kudo和Richardson于2018年在Google开发，是一个语言无关的分词工具包[^11^]。
- 它直接将训练语料视为原始Unicode字符序列，**没有预分词步骤**（即不基于空格提取单词），这使其可以应用于任何字符序列，实现了真正的语言无关性[^12^]。
- SentencePiece实现了两种主要算法：BPE（自底向上合并）和**Unigram Language Model**（自顶向下剪枝）[^13^]。
- Unigram方法从一个大型候选词汇表开始，迭代地剪除对语料对数似然贡献最小的token，使用Viterbi算法和EM算法计算和优化token概率[^14^]。
- SentencePiece提供可逆分词（reversibility），这对基于子词的语言模型至关重要[^15^]。

**关键文献**：
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing," EMNLP 2018[^11^]
- Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates," ACL 2018[^16^]

### 4. Byte-level BPE与未知字符问题

**核心发现**：
- GPT-2（Radford et al., 2019）引入了**Byte-level BPE（BBPE）**，直接在原始字节上应用BPE而非Unicode字符，使模型能够处理任何Unicode字符或二进制数据[^17^]。
- BBPE的核心优势是**完全消除了未知字符（UNK）问题**——因为任何文本都可以表示为256个字节的序列，词汇表永远不会有覆盖盲区[^18^]。
- GPT-2的词汇表大小为50,257（50,000个合并token + 256个字节token + 1个特殊token）[^19^]。
- BBPE的代价是序列长度显著增加——非拉丁文字（如中文、泰米尔文）的UTF-8编码需要2-3个字节/字符，导致这些语言的token序列更长[^20^]。

**关键文献**：
- Radford et al., "Language Models are Unsupervised Multitask Learners," 2019 (GPT-2)[^17^]

### 5. GPT系列Tokenizer的工程化演进

**核心发现**：
- **GPT-2**：r50k_base，~50K词汇，byte-level BPE，基础正则表达式模式[^21^]
- **GPT-3/3.5**：p50k_base，~50K词汇，改进的预分词模式[^21^]
- **GPT-4**：cl100k_base，~100,277词汇，重大改进的正则表达式模式，支持多语言和代码，引入FIM（Fill-in-the-Middle）特殊token[^22^]
- **GPT-4o/GPT-5**：o200k_base，~200,000词汇，2024年5月发布，显著改善非拉丁语言效率[^23^]

**GPT-4 tokenizer的关键改进**[^22^]：
- 大小写不敏感匹配 `(?i:...)` 处理大小写变体
- 数字仅匹配2位以上 `\p{N}{2,}`，防止超长数字序列合并为单一token
- 空白字符处理改进——GPT-4合并连续空格，而GPT-2不合并
- 特殊token扩展：`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` 支持代码补全

**o200k_base的多语言改进**[^23^][^24^]：
- 词汇表从100K翻倍至200K，非英语token数量显著减少
- 中文tokenization效率提升约1.4x，日文1.4x，韩文1.7x
- 印地语token数量减少高达79%，泰卢固语77%，泰米尔语74%
- 泰米尔文示例：同一句子从68 tokens降至21 tokens（3.2x压缩）[^24^]

**关键文献**：
- tiktoken library documentation, OpenAI[^22^]
- Various technical blogs on GPT-4o tokenization improvements[^23^][^24^]

### 6. Tokenizer对多语言模型的影响——Token预算不公平问题

**核心发现**：
- Petrov等人（2023）的开创性论文《Language Model Tokenizers Introduce Unfairness Between Languages》量化了低资源语言用户需要支付2-15倍更多费用的问题[^25^]。
- **"Tokenization Premium"**指标定义：`premium = n_tokens(language) / n_tokens(English)`。Premium为1.0表示与英语平等，2.0表示需要2倍token[^26^]。
- 在cl100k_base中，相同句子泰米尔语需要6倍于英语的token，印地语4.6倍，中文2倍[^27^]。
- **Byte Premium Effect**：字节级BPE中，UTF-8编码的非拉丁字符需要2-3个字节，导致序列长度隐含地惩罚非拉丁文字[^28^]。
- 即使明确为其他语言设计的模型（如CamemBERT法语模型、GottBERT德语模型），英语仍然获得最低的premium——这反映了训练数据中英语的主导地位和tokenizer的内在偏差[^25^]。
- 使用英语tokenizer处理非英语文本可导致高达68%的额外计算成本和10-15个百分点的零样本准确率下降[^29^]。

**关键文献**：
- Petrov et al., "Language Model Tokenizers Introduce Unfairness Between Languages," arXiv:2305.15425[^25^]
- Ahia et al., 相关后续研究[^26^]

### 7. 代码、数学、公式对Tokenizer的特殊挑战

**核心发现**：
- **代码Tokenization的语法不对齐问题**：Nie等人（2025）的TokDrift研究表明，BPE基于频率统计而非语法进行分词，导致token边界与编程语言语法token边界不对齐[^30^]。
- 即使是微小的格式变化（如空格编辑或标识符重命名）也可能导致模型输出发生重大变化。Qwen2.5-Coder-32B-Instruct在tokenization改变时预测变化率为6.09%，在某些单条重写规则下高达60%[^30^]。
- 层分析表明问题起源于早期嵌入层——子词分割未能捕获语法token边界[^30^]。
- **数学公式Tokenization挑战**：LaTeX符号和数学表达式包含大量特殊字符（\, _, ^, {, }），标准BPE往往将其分割为无意义的子词单元。数字作为token的处理也存在问题——长数字序列可能被拆分为多个子token，影响算术能力[^31^]。
- 代码预处理策略（如PLBART）将`\n`、缩进替换为`NEW_LINE`、`INDENT`、`DEDENT`token，因为这些是Python语法的一部分[^32^]。

**关键文献**：
- Nie et al., "TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar," arXiv:2510.14972[^30^]
- Ahmad et al., "CodeBPE: Investigating Subtokenization Options for LLM Pretraining on Source Code," arXiv:2308.00683[^32^]

### 8. Tokenizer-Free或原生字节建模的研究进展

**核心发现**：
- **Tokenizer-free架构**旨在完全消除固定的词汇表分词步骤，直接在原始字节或字符上操作[^33^]。主要分类：

**（1）纯字节级建模**：
- **ByT5**（Xue et al., 2022）：处理UTF-8字节序列，无需显式分词，在多语言和噪声鲁棒性上表现优异[^34^]
- **MambaByte**（Wang et al., 2024）：基于选择性状态空间架构的无token模型，在语言建模任务上优于subword和字节级Transformer[^35^]

**（2）启发式分块的层次建模**：
- **MEGABYTE**（Yu et al., 2023）：将字节序列分为固定大小的patch，全局模型处理patch间关系，局部模型处理patch内字节，实现百万字节序列的次二次方自注意力[^36^]
- **SpaceByte**（Slagle, 2024）：在空格字符后应用更大的Transformer块，匹配subword Transformer性能[^37^]

**（3）动态分块的层次建模**：
- **Byte Latent Transformer (BLT)**（Pagnoni et al., 2024, Meta）：基于**熵的动态patching**——小字节级语言模型计算下一个字节的熵，patch边界出现在下一个字节最难预测的位置。简单可预测区域（常见词）获得大patch，复杂区域（罕见词、代码、数字）获得小patch[^38^]
- BLT在8B参数规模上**匹配Llama 3性能**，同时使用多达**50%更少的推理FLOP**[^38^]
- **H-Net**（Hwang et al., 2025）：使用相邻表示的余弦相似度决定分块，端到端学习[^39^]
- **Fast BLT (BLT-D)**（2026）：结合扩散目标实现字节级并行生成，加速推理[^40^]

**关键文献**：
- Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens," arXiv:2412.09871[^38^]
- Xue et al., "ByT5: Towards a token-free future with pre-trained byte-to-byte models," TACL 2022[^34^]
- Yu et al., "MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers," NeurIPS 2023[^36^]
- Wang et al., "MambaByte: Token-free Selective State Space Model," COLM 2024[^35^]

### 9. 词表大小对训练效率和上下文长度的影响

**核心发现**：
- **词汇表大小的快速增长**：2023-2026年间词汇表大小增长了约8倍[^41^]：
  - Llama 2 (2023): 32K SentencePiece
  - Llama 3 (2024): 128K tiktoken-style
  - GPT-4o (2024): ~200K o200k_base
  - Gemma 3 (2025): 262K SentencePiece

- **压缩与泛化的权衡**[^42^]：
  - 太小的词汇表导致碎片化或过于细粒度的token，序列更长，语义表示下降
  - 太大的词汇表引入冗余，增加内存使用和模型开销
  - Tao et al. (NeurIPS 2024) 发现词汇表大小与训练损失之间存在对数线性关系

- **嵌入矩阵的内存瓶颈**：大词汇表导致大型嵌入矩阵和LM head，增加推理时的计算和内存开销。对于资源受限的部署，这成为可扩展性瓶颈[^43^]。

- **最优嵌入学习率与词汇表大小的关系**：Hayou和Liu（2025）的研究表明，当词汇表大小远大于嵌入维度时（m >> d），μP参数化的嵌入学习率不再最优。最优的嵌入LR/隐藏层LR比率应按`sqrt(d)`而非`d`缩放[^44^]。

- **词汇表大小应该是模型大小的函数**：Llama 2的32K词汇表对于7B模型是最优的，但对于70B变体，计算最优的词汇表应至少为216K（实际使用的7倍大）[^41^]。

**关键文献**：
- Tao et al., NeurIPS 2024 (词汇表大小scaling law)[^41^]
- Hayou & Liu, "Optimal Embedding Learning Rate in LLMs," arXiv:2506.15025[^44^]

### 10. 2024-2025 Tokenizer设计的最新趋势

**核心发现**：

**(a) Superword Tokenization——跨越空白的合并**：
- **SuperBPE**（Liu et al., 2025）：引入预分词课程——第一阶段学习subword（标准BPE），第二阶段学习跨越空白的superword。在200K固定词汇表大小下比BPE减少33%的token[^45^]
- **BoundlessBPE**（Schmidt et al., 2025）：独立提出放松预分词约束，将完整的pretoken合并为superwords。token频率分布更均匀（Rényi效率+21%），压缩提升约20%[^46^]
- 下游任务准确率平均提升4.0%（SuperBPE在30个下游任务上），MMLU提升8.2%[^45^]

**(b) 零样本Tokenizer迁移**：
- **ZeTT**（Minixhofer et al., 2024）：使用超网络将训练好的LLM迁移到新tokenizer，大幅降低为多语言工作负载重新tokenize开源模型的成本[^47^]

**(c) 词汇表扩展领域自适应**：
- 为特定语言或领域（法律、医疗、代码）扩展词汇表成为常见做法。例如，Youtu-LLM从o200k词汇表出发，保留前100K ASCII token，然后训练中文特定token和代码/数学专用token[^48^]

**(d) 推理时自适应词汇**：
- **zip2zip**（2025）：首个实现推理时动态词汇扩展的方法，基于输入上下文构建新token，无需重新训练或修改tokenizer[^49^]

**(e) 分词器公平性的量化与缓解**：
- 多语言Tokenizer的公平性研究成为活跃领域。通过为每种语言确定最优词汇表大小、使用SuperBPE跨越空白合并、添加缺失的Unicode字符token等方法来降低token premium[^50^]

**关键文献**：
- Liu et al., "SuperBPE: Superword Tokenization," 2025[^45^]
- Schmidt et al., "BoundlessBPE," 2025[^46^]
- Minixhofer et al., "ZeTT: Zero-Shot Tokenizer Transfer," arXiv:2405.07883[^47^]

---

## 二、重要论文和来源

### 经典基础论文

| 论文 | 作者 | 年份 | 会议/来源 | 核心贡献 |
|------|------|------|-----------|----------|
| "Neural Machine Translation of Rare Words with Subword Units" | Sennrich et al. | 2016 | ACL | 将BPE引入NLP[^1^] |
| "Japanese and Korean Voice Search" | Schuster & Nakajima | 2012 | Google | WordPiece算法[^6^] |
| "Google's Neural Machine Translation System" | Wu et al. | 2016 | arXiv | WordPiece在GNMT中的应用[^10^] |
| "SentencePiece" | Kudo & Richardson | 2018 | EMNLP | 语言无关分词工具包[^11^] |
| "Language Models are Unsupervised Multitask Learners" | Radford et al. | 2019 | OpenAI | GPT-2 Byte-level BPE[^17^] |

### 2020-2023重要研究

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|----------|
| "Byte Pair Encoding is Suboptimal for Language Model Pretraining" | Bostrom & Durrett | 2020 | EMNLP Findings | BPE在预训练中的次优性[^51^] |
| "ByT5: Towards a token-free future" | Xue et al. | 2022 | TACL | 字节级T5模型[^34^] |
| "CANINE: Pre-training an Efficient Tokenization-Free Encoder" | Clark et al. | 2022 | TACL | Unicode codepoint级别编码器[^52^] |
| "MEGABYTE" | Yu et al. | 2023 | NeurIPS | 百万字节序列多尺度Transformer[^36^] |
| "Language Model Tokenizers Introduce Unfairness Between Languages" | Petrov et al. | 2023 | arXiv | 多语言Tokenizer不公平性量化[^25^] |

### 2024-2025前沿研究

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|----------|
| "Byte Latent Transformer: Patches Scale Better Than Tokens" | Pagnoni et al. (Meta) | 2024 | arXiv | 基于熵的动态字节patching[^38^] |
| "MambaByte: Token-free Selective State Space Model" | Wang et al. | 2024 | COLM | 状态空间字节模型[^35^] |
| "SuperBPE: Superword Tokenization" | Liu et al. | 2025 | arXiv | 跨越空白的superword[^45^] |
| "BoundlessBPE" | Schmidt et al. | 2025 | arXiv | 放松预分词约束[^46^] |
| "TokDrift" | Nie et al. | 2025 | arXiv | 代码Tokenizer语法不对齐[^30^] |
| "Fast Byte Latent Transformer" | (BLT-D) | 2026 | arXiv | 字节级扩散解码[^40^] |
| "Improbable Bigrams Expose Vulnerabilities" | | 2024 | arXiv | 不完整token的安全漏洞[^53^] |
| "Reducing Tokenization Premiums for Low-Resource Languages" | | 2025 | arXiv | 降低低资源语言token溢价[^50^] |

---

## 三、趋势和信号

### 趋势1：词汇表大小持续扩张

从大模型历史看，词汇表大小呈指数增长。2023年主流模型（Llama 2）使用32K词汇表，到2025年Gemma 3已达262K。这一趋势受两个因素驱动：
1. **多语言覆盖需求**：更大词汇表可以为非拉丁文字分配更多token slot，减少token premium
2. **计算效率**：更大的词汇表产生更短的序列长度，减少自注意力计算

### 趋势2：从固定Tokenizer向动态/自适应Tokenizer演进

传统BPE在训练前构建固定词汇表。新兴方向包括：
- **动态patching（BLT）**：根据输入内容自适应调整分块大小
- **推理时自适应（zip2zip）**：根据输入上下文动态构建新token
- **领域自适应词汇扩展**：针对特定领域或语言扩展现有tokenizer

### 趋势3：Tokenizer-free架构从概念走向实用

- BLT首次证明字节级模型可以在大规模（8B参数）上匹配BPE基线
- 字节级模型消除了UNK、glitch token、多语言不平等等问题
- 主要瓶颈是序列长度导致的注意力计算量，但层次化架构和状态空间模型正在缓解这一问题
- 未来可能在多模态场景（直接处理文件字节）中率先取代tokenization

### 趋势4：代码和多模态对Tokenizer设计提出新要求

- 代码需要语法感知的分词（TokDrift揭示了当前BPE的严重缺陷）
- 数学公式需要特殊处理（LaTeX符号、数字序列）
- 多模态输入（图像、音频的字节表示）推动tokenizer-free架构

### 趋势5：Tokenizer公平性成为核心设计考量

- 2023年后，所有主要模型发布都将多语言token效率作为关键指标
- o200k_base、Llama 3 128K等明确以改善非英语tokenization为设计目标
- 社区正在发展标准化的公平性度量（tokenization premium）和缓解方法

---

## 四、争议和冲突观点

### 争议1：BPE是否是最优选择？

- **支持BPE**：Bostrom & Durrett (2020) 的论文标题直接指出"BPE is Suboptimal"，认为BPE不能有效利用词汇空间[^51^]。WordPiece和Unigram在理论上可能更优
- **支持BPE的反驳**：实际上BPE仍是SOTA LLM的首选，因其内存需求低、实现简单、训练稳定。Ali et al. (2023) 发现对于英语任务，tokenizer选择影响可以忽略，但对多语言和翻译任务影响巨大[^54^]

### 争议2：大词汇表vs小词汇表

- **大词汇表支持者**：更大的词汇表缩短序列长度、改善多语言覆盖、降低推理FLOP
- **小词汇表支持者**：大词汇表增加嵌入矩阵内存、降低训练效率、可能导致低频token训练不足
- **Tao et al. (2024)** 的发现：词汇表大小与训练损失之间存在对数线性关系，存在计算最优的词汇表大小

### 争议3：Tokenizer-free架构能否取代BPE？

- **乐观派**（Meta BLT团队）：字节级模型在固定推理成本下展现出比tokenizer模型更好的scaling特性[^38^]
- **谨慎派**：字节级模型的序列长度仍然是主要瓶颈。虽然BLT在推理FLOP上有优势，但实际wall-clock时间优势尚不明确，且attention的quadratic复杂度对长序列仍是挑战

### 争议4：Tokenization Premium是否等于性能不平等？

- **Petrov et al. (2023)** 量化了token count的不公平性
- **批评者**指出：tokenization disparity不等于performance disparity。更高的premium可能反映形态学复杂性（如土耳其语的黏着性），而非tokenizer的缺陷
- **尚未验证**：高premium是否因果性地降低任务性能，这一经验性联系尚未被测量[^26^]

---

## 五、建议深入研究的领域

### 第6章素材：Tokenizer的核心问题（压缩与泛化）

**核心主题**：Tokenizer在LLM中的双重角色——文本压缩与语义泛化之间的张力

**建议深入方向**：
1. **压缩效率的度量**：fertility（每词token比率）、power-law deviation、Rényi efficiency等内在指标与下游任务表现的相关性
2. **词汇表构造的信息论视角**：将tokenization视为有损/无损压缩问题，探讨最优分割点的信息论界限
3. **Morphology-aware tokenization**：如何设计更好地尊重形态学边界的tokenizer

### 第7章素材：BPE、WordPiece、SentencePiece到Byte-level Tokenizer

**核心主题**：分词技术的完整演进链条及其工程实践

**建议深入方向**：
1. **BPE的贪心策略分析**：为什么贪心合并在实践中有效，理论上的次优性在何种条件下显著
2. **WordPiece的PMI标准vs BPE的频率标准**：何时PMI显著优于频率，BERT的经验教训
3. **SentencePiece的无预分词设计**：Unicode直接处理的优势、代价和边界情况
4. **Byte-level BPE的UTF-8编码问题**：字节premium效应对不同文字系统的影响量化
5. **各代GPT tokenizer的工程决策细节**：正则表达式模式的演变、特殊token的设计哲学

### 第8章素材：2023-2026 Tokenizer的新问题

**核心主题**：大模型时代tokenization面临的新挑战和新范式

**建议深入方向**：
1. **多语言Tokenizer公平性的系统性研究**：token premium的分解（形态学vs tokenizer偏差）、标准化评估协议
2. **代码和数学领域的专用tokenizer设计**：语法感知分词、数字处理、LaTeX符号
3. **Tokenizer-free架构的实用化路径**：BLT类方法的效率优化、与现有推理基础设施的兼容性
4. **Glitch token和安全漏洞**：不完整token的安全风险、自动化检测（GlitchMiner）、防御策略
5. **词汇表动态扩展和迁移**：ZeTT、zip2zip等方法如何在生产环境中部署
6. **Superword tokenization的前景**：SuperBPE和BoundlessBPE是否能成为下一代标准
7. **Tokenizer与模型训练动态的关系**：词汇表大小对学习率、收敛速度、最终性能的影响

---

## 引用索引

[^1^]: Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units," ACL 2016
[^2^]: arXiv:2506.01687, "StochasTok: Improving Fine-Grained Subword Understanding in LLMs"
[^3^]: arXiv:2506.01687, BPE作为SOTA LLM首选
[^4^]: arXiv:2508.04796, "Evaluation of Coding Schemes for Transformer-based Gene Sequence Modeling"
[^5^]: Gage, "A New Algorithm for Data Compression," The C Users Journal, 1994
[^6^]: Schuster & Nakajima, "Japanese and Korean Voice Search," 2012
[^7^]: arXiv:2402.18376, "Tokenization Is More Than Compression"
[^8^]: arXiv:2409.14579, "Medical Concept Normalization in a Low-Resource Setting"
[^9^]: arXiv:2004.03720, "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
[^10^]: Wu et al., "Google's Neural Machine Translation System," 2016
[^11^]: Kudo & Richardson, "SentencePiece," EMNLP 2018
[^12^]: arXiv:2106.07540, "Evaluating Various Tokenizers for Arabic Text Classification"
[^13^]: arXiv:2406.19223, "Tokenizer-Free Generative LLMs via Sparse Representations"
[^14^]: arXiv:2512.12641, "Which Pieces Does Unigram Tokenization Really Need?"
[^15^]: arXiv:2009.12534, "iNLTK: Natural Language Toolkit for Indic Languages"
[^16^]: Kudo, "Subword Regularization," ACL 2018
[^17^]: Radford et al., "Language Models are Unsupervised Multitask Learners," 2019
[^18^]: arXiv:2604.17814, "Understanding Secret Leakage Risks in Code LLMs"
[^19^]: tiktoken library, OpenAI
[^20^]: arXiv:2505.24689, "Structured Encoding for Robust Multilingual Pretokenization"
[^21^]: Multiple sources on GPT-2/3 tokenizer
[^22^]: fast.ai blog on GPT tokenizer; tiktoken openai_public.py source
[^23^]: futureagi.com blog; OpenAI documentation
[^24^]: njkumar.com blog on multilingual token compression
[^25^]: Petrov et al., "Language Model Tokenizers Introduce Unfairness," arXiv:2305.15425
[^26^]: anuadesina.com, "The Tokenization Tax"
[^27^]: iotdigitaltwinplm.com, "LLM Tokenization Deep Dive"
[^28^]: Petrov et al., 2023; arXiv:2505.24689
[^29^]: emergentmind.com, "Multilingual GPT-Scale Tokenizers"
[^30^]: Nie et al., "TokDrift," arXiv:2510.14972
[^31^]: Various sources on mathematical tokenization challenges
[^32^]: Ahmad et al., "CodeBPE," arXiv:2308.00683
[^33^]: arXiv:2603.03583, "ByteFlow: Language Modeling through Adaptive Byte Compression"
[^34^]: Xue et al., "ByT5," TACL 2022
[^35^]: Wang et al., "MambaByte," COLM 2024
[^36^]: Yu et al., "MEGABYTE," NeurIPS 2023
[^37^]: Slagle, "SpaceByte," NeurIPS 2024
[^38^]: Pagnoni et al., "Byte Latent Transformer," arXiv:2412.09871
[^39^]: Hwang et al., "H-Net," 2025
[^40^]: arXiv:2605.08044, "Fast Byte Latent Transformer"
[^41^]: letsdatascience.com, "Tokenization Deep Dive"
[^42^]: arXiv:2507.22543, "Pre-trained Models Perform the Best When Token Distributions Follow Zipf's Law"
[^43^]: arXiv:2508.15229, "VocabTailor: Dynamic Vocabulary Selection"
[^44^]: Hayou & Liu, "Optimal Embedding Learning Rate in LLMs," arXiv:2506.15025
[^45^]: Liu et al., "SuperBPE," 2025
[^46^]: Schmidt et al., "BoundlessBPE," arXiv:2504.00178
[^47^]: Minixhofer et al., "ZeTT," arXiv:2405.07883
[^48^]: arXiv:2512.24618, "Youtu-LLM"
[^49^]: arXiv:2506.01084, "zip2zip: Inference-Time Adaptive Vocabularies"
[^50^]: arXiv:2601.13328, "Reducing Tokenization Premiums for Low-Resource Languages"
[^51^]: Bostrom & Durrett, "BPE is Suboptimal," EMNLP 2020 Findings
[^52^]: Clark et al., "CANINE," TACL 2022
[^53^]: arXiv:2410.23684, "Improbable Bigrams Expose Vulnerabilities"
[^54^]: Ali et al., "Tokenizer choice for LLM training," arXiv:2310.08754
