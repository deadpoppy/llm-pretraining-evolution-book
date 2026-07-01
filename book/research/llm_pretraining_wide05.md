# 深度研究：预训练数据配比的演进（第13-16章素材）

> **研究范围**：覆盖大模型预训练数据的选择、清洗、配比策略的演进，从2020年The Pile到2026年的最新趋势
> **研究日期**：2025年7月
> **核心主题**：数据配比从"经验主义"走向"可预测、可优化"的技术革命

---

## 第13章：预训练数据的基本构成和清洗

### 13.1 大规模预训练语料库的家族图谱

#### 13.1.1 早期奠基性语料库（2020-2021）

**The Pile**（EleutherAI, 2020）是预训练数据工程的里程碑 [^1^]。作为一个825GB的英语文本语料库，它由22个高质量子数据集精心组合而成，包括Pile-CC（Common Crawl子集，227GB）、PubMed、Books3、OpenWebText2、ArXiv、GitHub、StackExchange、FreeLaw等 [^1^][^2^]。The Pile的设计理念强调**领域多样性**，让模型同时学习学术、法律、编程、创意写作等不同风格。GPT-Neo、GPT-NeoX、Pythia等早期开源模型都基于The Pile训练 [^2^]。

**C4（Colossal Clean Crawled Corpus）**（Google, 2020）是为T5模型创建的约750GB英语文本数据集，通过启发式规则从Common Crawl清洗而来。它移除了JavaScript代码片段、法律模板条款（"Terms of Conditions"）和低质量内容 [^3^]。C4证明了一件事：**经过严格清洗的web数据可以作为高质量预训练语料**。

**MassiveText**（DeepMind, 2021）是为Gopher训练的集合，包含2.35亿文档（约10.5TB文本），但实际只采样了300B tokens（12.8%）。其子集MassiveWeb是通过严格的清洗流程（质量过滤、去重、训练-测试重叠移除）从Common Crawl中提取的高质量web文本 [^4^]。

#### 13.1.2 第二代高质量语料库（2022-2023）

**RefinedWeb**（TII, 2023）是一个约5T tokens的去重Common Crawl数据集，用于Falcon系列模型。其核心论点是：**经过激进MinHash去重的纯web数据，性能可以超越混合了精心策划来源（如书籍、Wikipedia）的语料库** [^5^]。这一发现挑战了传统"混合多样来源"的直觉。

**SlimPajama**（Cerebras, 2023）对RedPajama应用MinHash去重，移除了49%的内容，展示了大规模语料内部的重复程度 [^6^]。

**ROOTS**（BigScience, 2022）是为BLOOM模型训练的多语言语料，包含46种自然语言和13种编程语言，共约366B tokens。ROOTS开创性地提出了**语言平等主义**的数据收集理念，强调不应让英语主导所有训练数据 [^7^]。

#### 13.1.3 第三代开放语料库（2024-2025）

**FineWeb**（Hugging Face, 2024年5月）成为2024年后默认的开放预训练数据集。它从96个月度Common Crawl快照（2013-2024）中提取了**15万亿tokens**的英语web语料 [^8^]。FineWeb-Edu是其质量分类器过滤变体，仅1.3T tokens但在固定计算量下表现优于完整的15T原始数据集 [^8^]。

**FineWeb-2**（Hugging Face, 2024）将高质量web数据扩展到**1000+种语言**，包含350亿罗马尼亚语tokens等多语言内容 [^9^]。

**DataComp-LM（DCLM）**（Apple/AI2等, 2024）提供了从Common Crawl中提取的240万亿tokens标准化语料池，以及固定的预训练配方，用于系统性地基准测试数据筛选策略 [^10^]。

**Nemotron-CC**（NVIDIA, 2024）和商业级的Nemotron-Pretraining Dataset提供了约6.3T tokens的精选预训练文本 [^11^]。

| 数据集 | 规模 | 维护者 | 年份 | 特点 |
|---|---|---|---|---|
| The Pile | 825GB / ~300B tokens | EleutherAI | 2020 | 22个高质量来源，多样性 |
| C4 | 750GB | Google | 2020 | 启发式清洗的CC |
| MassiveText | 10.5TB | DeepMind | 2021 | Gopher训练数据 |
| RefinedWeb | 5T tokens | TII | 2023 | 激进去重web数据 |
| ROOTS | 366B tokens | BigScience | 2022 | 46种自然语言 |
| FineWeb | 15T tokens | HuggingFace | 2024 | 96个CC快照 |
| FineWeb-2 | 多语言 | HuggingFace | 2024 | 1000+语言 |
| DCLM | 240T tokens池 | Apple/AI2 | 2024 | 基准测试平台 |
| Nemotron-CC | 6.3T tokens | NVIDIA | 2024 | 商业级质量 |

[^1^]: Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," 2020.
[^2^]: EleutherAI, The Pile documentation, https://pile.eleuther.ai/
[^3^]: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," T5 paper, 2020.
[^4^]: Rae et al., "Scaling Language Models: Methods, Analysis & Insights from Training Gopher," DeepMind, 2021.
[^5^]: Penedo et al., "The RefinedWeb Dataset for Falcon LLM," 2023.
[^6^]: Soboleva et al., "SlimPajama: A 627B token cleaned and deduplicated version of RedPajama," Cerebras, 2023.
[^7^]: BigScience Workshop, "A 176B-Parameter Open-Access Multilingual Language Model," BLOOM paper, 2022.
[^8^]: Penedo et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," Hugging Face, 2024.
[^9^]: FineWeb-2 technical documentation, Hugging Face, 2024.
[^10^]: Li et al., "DataComp-LM: In search of the next generation of training sets for language models," 2024.
[^11^]: Su et al., "Nemotron-CC: Transforming Common Crawl into a Customized Large-Scale High-Quality Dataset," NVIDIA, 2024.

### 13.2 数据清洗：从启发式到机器学习

#### 13.2.1 数据清洗的三层架构

现代预训练数据清洗通常采用三层架构 [^12^][^13^]：

**第一层：文档级启发式过滤**
- URL黑名单过滤（移除NSFW域名、已知垃圾网站）
- 长度过滤（移除过短或过长的文档）
- 符号比例过滤（如字符重复率、行尾标点比例）
- 语言检测（使用FastText等工具过滤非目标语言）
- 规则匹配（移除包含"Terms of Conditions"等法律模板的内容）

**第二层：去重**
- **MinHash + LSH**：5-gram shingles的MinHash签名（通常H=112个哈希函数，14个band，每个band 8个hash），Jaccard相似度阈值约0.8 [^6^][^5^]
- **精确子串去重**：使用后缀数组移除50+ tokens的重复片段（FineWeb采用）
- **SemDeDup**：基于模型的语义去重，捕获语义相似但表述不同的文档，但计算成本高 [^12^]
- **Cross-source去重**：MixMinMatch方法利用跨语料库重复作为质量信号——被多个独立管道保留的内容更可能是高质量文本 [^14^]

**第三层：基于模型的质量评分**
- **困惑度过滤**：使用在Wikipedia上训练的KenLM语言模型计算文档困惑度，过滤掉高于阈值的文档 [^7^]
- **质量分类器**：FineWeb-Edu使用Llama-3-70B标注的教育价值数据训练轻量级评分器 [^8^]
- **fastText分类器**：DCLM-Baseline使用fastText质量分类器，训练数据来自OpenHermes和r/ExplainLikeImFive等高质量子reddit [^10^]
- **GPT-4作为金标准**：SIEVE方法使用GPT-4o训练质量过滤器，在DataComp-LM上超越了DCLM基线 [^15^]

#### 13.2.2 去重的重要发现

去重不仅是一个工程优化，它对模型质量有根本性影响：

1. **训练效率提升**：Tirumala等（2023）证明去重结合多样性选择可提升20%训练效率 [^16^]
2. **记忆化减少**：Lee等（2022）证明去 duplicated 数据减少模型对重复内容的记忆化，改善泛化 [^16^]
3. **纯web数据 > 混合语料**：Falcon团队证明，经过激进MinHash去重的纯web数据，性能可超越混合了书籍和Wikipedia的The Pile [^5^]
4. **跨源一致性作为质量信号**：MixMinMatch发现，跨主要阿拉伯web语料库超过40%的tokens是重复的，这种冗余可以作为免费的质量过滤器 [^14^]

[^12^]: Albalak et al., "A Survey on Data Selection for Language Models," 2024.
[^13^]: Wettig et al., "QuRating: Quality ratings for pretraining data," 2024.
[^14^]: "Mix, MinHash, and Match: Cross-Source Agreement for Multilingual Pretraining Datasets," 2025.
[^15^]: "GPT-4o as the Gold Standard: A Scalable and General Purpose Approach to Filter Language Model Pretraining Data," 2024.
[^16^]: Lee et al., "Deduplicating Training Data Makes Language Models Better," 2022; Tirumala et al., 2023.

---

## 第14章：2020-2022大规模语料堆叠阶段

### 14.1 "数据越多越好"的时代（2020-2022）

#### 14.1.1 GPT-3与大规模Common Crawl的验证

2020年GPT-3的发布标志着**"规模即一切"**（Scale is All You Need）范式的确立。GPT-3在约300B tokens上训练，数据来源包括过滤后的Common Crawl（60%）、WebText2（22%）、Books1（8%）、Books2（8%）和Wikipedia（3%） [^17^]。这验证了：

1. 大规模web数据可以直接用于预训练
2. 高质量来源（如Wikipedia、书籍）需要upweighting
3. 混合不同来源比单一来源效果更好

#### 14.1.2 Gopher的MassiveText：系统化的数据配比实验

DeepMind的Gopher（280B参数，2021年）在MassiveText上的训练是最早**系统性地调整数据采样比例**的研究之一 [^4^]。MassiveText包含以下组成部分：

- **MassiveWeb**：DeepMind自行策划的web文本集合，经过严格清洗
- **C4**：Google的清洗版Common Crawl
- **Books**：书籍数据
- **News**：新闻文章
- **GitHub**：代码数据
- **Wikipedia**：百科全书文本

Gopher团队发现，**调整各子集的采样比例可以显著影响下游性能**。例如，高质量的web文本（MassiveWeb）对下游任务的帮助超过了C4。Gopher在300B tokens上训练，仅为MassiveText中总tokens的12.8%，说明数据选择的重要性 [^4^]。

#### 14.1.3 Chinchilla的启示：数据规模与模型规模同等重要

DeepMind的Chinchilla（2022年）提出了著名的**缩放法则**（Scaling Laws）：在固定计算预算下，模型参数量和数据量应等比例增长 [^18^]。Chinchilla（70B参数）在1.4T tokens上训练，远超当时同类模型的训练量。这一发现推动了一个关键转变：**数据量不再被视为无限资源，而是需要精心规划和优化的核心要素**。

Chinchilla的MassiveText组成与Gopher类似，但训练tokens增加了约4.7倍，这一突破使得Chinchilla在远小于Gopher（280B）的参数量下取得了更优性能。

#### 14.1.4 BLOOM与多语言数据的引入

BLOOM（BigScience, 2022）是一个176B参数的多语言模型，在ROOTS（366B tokens）上训练 [^7^]。ROOTS的构成是预训练数据工程史上的重要里程碑：

- 46种自然语言 + 13种编程语言
- 明确反对英语中心主义
- 采用**温度采样**（temperature sampling）来平衡不同语言的数据量
- 低资源语言通过温度采样获得过采样（up-sampling）

BLOOM引入了**"多语言诅咒"**（Curse of Multilinguality）的概念：在固定模型容量下，增加语言数量最初有利于低资源语言的跨语言迁移，但超过某个点后，单语和跨语言性能都开始下降 [^19^]。

#### 14.1.5 PaLM与代码数据的初步探索

Google的PaLM（2022年，540B参数）在780B tokens上训练，其训练数据混合包含了web数据、对话数据、书籍、Wikipedia和**GitHub代码** [^20^]。PaLM展示了代码数据对模型能力的显著提升，特别是在推理任务上。这一发现为后续**"代码数据提升推理能力"**的研究奠定了基础。

### 14.2 这一阶段的核心特征

**特征一：经验主义主导**。数据配比主要依赖研究者的直觉和手动调参。例如，Gopher团队"调整采样比例以最大化下游性能"，但缺乏系统性方法论 [^4^]。

**特征二：高质量来源upweighting**。Wikipedia、书籍等被视为"高质量"来源，通常被过采样。例如GPT-3的Wikipedia占比3%，远高于其在web中的自然比例。

**特征三：数据规模竞赛**。各研究机构竞相收集更大的语料库——从GPT-3的300B到PaLM的780B，再到LaMDA的1.56T。数据质量虽被关注，但规模增长是主要驱动。

**特征四：清洗是事后补救**。数据清洗被视为必要的预处理步骤，而非训练策略的核心组成部分。

[^17^]: Brown et al., "Language Models are Few-Shot Learners," GPT-3 paper, NeurIPS 2020.
[^18^]: Hoffmann et al., "Training Compute-Optimal Large Language Models," Chinchilla paper, 2022.
[^19^]: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale," XLM-R paper, 2020.
[^20^]: Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," Google, 2022.

---

## 第15章：2023-2024数据配比成为核心技术

### 15.1 数据配比意识的觉醒

#### 15.1.1 LLaMA系列：数据配比的商业化实践

**LLaMA-1**（Meta, 2023年2月）的训练数据构成标志着数据配比的商业化成熟。1.4T tokens来自公开数据源，经过精心的比例调配 [^21^]：

- Common Crawl（67%）
- C4（15%）
- GitHub（4.5%）
- Wikipedia（4.5%）
- Books（4.5%）
- ArXiv（2.5%）
- StackExchange（2%）

**LLaMA-2**在2T tokens上训练，数据构成类似但规模更大。

**LLaMA-3**（Meta, 2024年4月）是数据配比的巅峰之作，在**15T+ tokens**上训练，是LLaMA-2的7倍。其数据构成经过scaling law实验优化 [^22^][^23^]：

- **50% 通用知识**（web数据）
- **25% 数学与推理**
- **17% 代码**
- **8% 多语言数据**（30+语言）

LLaMA-3的关键创新包括：
1. **代码数据增加4倍** compared to LLaMA-2
2. **专门的数学、推理和代码提取pipeline**，使用prompt-tuned模型从STEM内容中提取
3. **多语言质量排序**：使用多语言LLaMA-2分类器对176种语言进行质量评分
4. **退火阶段**（Annealing）：在40M tokens上的退火阶段使GSM8k提升24.0%，MATH提升6.4% [^22^]

LLaMA-3.1（2024年7月）进一步增加到15.6T tokens，并包含2500万合成生成样本 [^23^]。

#### 15.1.2 Falcon：去重web数据的胜利

Falcon-180B（TII, 2023年）几乎完全在**RefinedWeb**（5T tokens的去重Common Crawl）上训练。其核心发现是：**经过激进去重和过滤的纯web数据，不需要传统的书籍、Wikipedia等"高质量"来源混合，就能达到顶级性能** [^5^]。这一发现动摇了传统数据配比的信念体系。

### 15.2 DoReMi：用小模型优化大模型的数据配比

DoReMi（Xie et al., 2023）是数据配比从经验走向科学的关键突破 [^24^]。

#### 15.2.1 DoReMi的核心机制

DoReMi（Domain Reweighting with Minimax Optimization）采用了一个 elegant 的两阶段策略：

**阶段1**：训练一个小型**代理模型**（proxy model，如280M参数），使用**Group DRO**（Distributionally Robust Optimization）在不同领域上训练。Group DRO的目标是**最小化最坏情况下的领域超额损失**（excess loss relative to a reference model），这自动产生了**领域权重**（domain weights）。

**阶段2**：使用这些领域权重重新采样数据集，然后训练一个更大的目标模型（如8B参数，比代理模型大30倍）。

#### 15.2.2 DoReMi的关键成果

- 在The Pile上，DoReMi将所有领域的困惑度都提高了，即使它**降低**了某些领域的权重
- 平均few-shot下游准确率提升**6.5个百分点**
- 达到基线准确率所需的训练步数减少**2.6倍**
- 在GLaM数据集上，DoReMi（不了解下游任务）匹配了基于下游任务调优的领域权重 [^24^]

#### 15.2.3 DoReMi的意义

DoReMi证明了一个革命性观点：**小代理模型的训练可以用来预测并优化大模型的数据配比**。这开启了"小规模实验指导大规模训练"的新范式，极大地降低了数据配比的实验成本。

### 15.3 DataComp-LM：数据筛选的奥林匹克

DataComp-LM（Li et al., 2024）是一个**正式的竞赛/基准**，用于系统性地比较不同的预训练数据筛选策略 [^10^]。

#### 15.3.1 DCLM的设计

DCLM包含以下核心组件：
- **DCLM-Pool**：240T tokens从Common Crawl提取的标准化语料池
- **固定预训练配方**：标准化数据加载、训练超参数、评估任务
- **53个下游NLU任务**用于评估
- **多尺度实验**：从400M到7B参数的模型规模

#### 15.3.2 DCLM的关键发现

- 最佳数据筛选策略相比随机选择，在MMLU上提升了**6.6个百分点**
- DCLM-Baseline采用fastText质量分类器，是强劲的起点
- **模型基础的筛选**（如FineWeb-Edu的教育价值分类器）显著优于规则基线
- 在1.64T tokens的数据池上，不同筛选策略的表现差异巨大，证明数据选择是核心竞争力

DCLM的更大意义在于：**它将数据工程从"暗知识"（tacit knowledge）转变为可公开比较和复制的科学**。

### 15.4 RegMix：数据配比即回归

RegMix（Liu et al., 2024年7月）提出了另一种数据配比优化方法 [^25^]：

1. **训练大量小代理模型**（512个1M参数模型），每个在随机数据配比上训练1B tokens
2. **拟合回归模型**（如LightGBM），以数据配比为特征，验证损失为目标
3. **模拟搜索最优配比**，然后用于大规模训练

RegMix的核心假设是**"秩不变性"**（rank invariance）：数据配比对性能影响的**相对排序**在不同模型规模和训练长度下保持一致。

RegMix的关键发现 [^25^]：
- 数据配比对性能的影响可高达**14.6%**的单任务性能差异
- **通用web语料**（如Common Crawl）而非Wikipedia等传统"高质量"来源，与下游性能的正相关性最强
- 领域间的交互复杂且常常**违反直觉**，凸显自动方法的必要性
- 使用仅2%的计算量就匹配或超越了DoReMi

### 15.5 代码数据进入预训练主流

#### 15.5.1 代码数据对推理能力的提升

多项研究在2023-2024年确认了代码数据对LLM推理能力的显著提升 [^26^][^27^]：

1. **预训练阶段混合代码和文本**可以显著增强通用推理能力，且几乎不会对其他任务产生负面迁移 [^27^]
2. **CodeLlama-34B**（在代码上继续预训练的LLaMA）在博弈论推理上达到了GPT-3.5-turbo的水平，显著超越了更大的LLaMA-2-70B [^28^]
3. **StarCoder2-15B**在数学和代码推理基准上超越了比它大两倍多的CodeLlama-34B [^29^]
4. 在指令微调阶段添加代码数据也能增强推理能力，不同模型家族的增益从3.6%到34.7%不等 [^30^]

#### 15.5.2 大型代码语料库的建设

- **The Stack v1**（BigCode, 2022）：3.1TB许可源代码，30种编程语言
- **The Stack v2**（BigCode/Software Heritage, 2024）：67.5TB原始数据，619种编程语言，经过清洗后得到900B+ tokens [^29^]
- **StarCoder2**在3.3-4.3T tokens上训练，远超Chinchilla最优建议的数量 [^29^]

#### 15.5.3 为什么代码提升推理能力？

研究者们提出了多个假设：
1. **结构化思维**：代码要求精确的、分步骤的逻辑推理
2. **低歧义性**：相比自然语言，代码的逻辑一致性更高
3. **执行反馈**：代码可以通过执行验证正确性，形成"ground truth"
4. **思维链（Chain-of-Thought）的原型**：代码注释和文档字符串本质上就是思维链的一种形式

### 15.6 多语言数据配比的深化

#### 15.6.1 温度采样与数据平衡

多语言模型的标准做法是**温度采样**（temperature sampling）：对语言l的采样概率进行重新加权 [^19^]：

```
p_l ~ (n_l)^(1/T) / sum((n_k)^(1/T))
```

其中n_l是语言l的自然比例，T是温度。T=1是自然分布，T>1使低资源语言获得更多采样。

#### 15.6.2 "多语言诅咒"的再审视

2025年的FineWeb-2研究重新审视了多语言诅咒，发现了令人意外的结果 [^19^]：

1. **增加语言数量不一定损害英语性能**——关键在于数据分布而非语言数量
2. 在固定总预算下，从25种语言扩展到400种，英语验证损失几乎不变（自然分布下）
3. **"诅咒"可以转化为"收益"**：高质量过滤后的多语言模型在同等tokens下优于单语模型
4. 低资源语言的问题是**质量和数量不足**，而非语言数量本身 [^31^]

这些发现表明，**高质量的多语言数据配比可以打破传统的"多语言诅咒"**。

### 15.7 MATES：模型感知的数据选择

MATES（Yu et al., 2024）引入了**数据影响模型**（data influence model）的概念 [^32^]：

- 预训练模型的数据偏好是**动态演化**的，而非静态的
- 通过局部探测预训练模型来收集"神谕"数据影响信号
- 微调一个小型影响模型来近似这些信号
- 影响模型预测整个语料库的数据影响，为下一预训练阶段选择最具影响力的数据

MATES在410M和1B模型上训练，相比随机选择显著提升了下游任务性能，且将state-of-the-art数据选择方法的收益翻倍。

### 15.8 合成数据的初步应用

#### 15.8.1 生成器驱动的方法

- **Tiny Stories**（Eldan & Li, 2023）：使用GPT-3.5/4生成合成数据训练小型模型
- **Phi-1/1.5**（Microsoft, 2023）：使用"教科书质量"的合成数据训练，1.3B参数模型超越更大模型
- **Cosmopedia**（Hugging Face, 2024）：开源的数十亿tokens合成教科书数据集

#### 15.8.2 源重述（Source Rephrasing）范式

WRAP（Web Rephrase Augmented Pre-training, Maini et al., 2024）提出了不同的方法 [^33^]：
- 不依赖大型模型从零生成数据
- 而是使用**小型LLM将现有web数据重述**为更高质量的格式
- 这种方法计算成本更低，同时保持多样性和覆盖度
- 通过策略性重述将预训练速度提升**3倍以上**

[^21^]: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," Meta, 2023.
[^22^]: GroverGPT technical paper, citing LLaMA-3 data composition; Meta Llama 3 Model Card.
[^23^]: Meta, "The Llama 3 Herd of Models," 2024.
[^24^]: Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining," Google, NeurIPS 2023.
[^25^]: Liu et al., "RegMix: Data Mixture as Regression for Language Model Pre-training," 2024.
[^26^]: Roziere et al., "Code Llama: Open Foundation Models for Code," Meta, 2023.
[^27^]: "At Which Training Stage Does Code Data Help LLMs Reasoning?" ICLR 2024 submission.
[^28^]: "GTBench: Uncovering the Strategic Reasoning Limitations of LLMs via Game-Theoretic Evaluations," 2024.
[^29^]: Lozhkov et al., "StarCoder2 and The Stack v2: The Next Generation," 2024.
[^30^]: "Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning," 2024.
[^31^]: "Revisiting Multilingual Data Mixtures in Language Model Pretraining," 2025.
[^32^]: Yu et al., "MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models," NeurIPS 2024.
[^33^]: Maini et al., "WRAP: Web Rephrase Augmented Pre-training," 2024; Su et al., "Nemotron-CC," 2024.

---

## 第16章：2025-2026数据配比的新趋势

### 16.1 数据配比方法的演进：从静态到动态

#### 16.1.1 离线方法（静态配比）

| 方法 | 年份 | 核心机制 | 计算成本 |
|---|---|---|---|
| DoReMi | 2023 | Group DRO代理模型 | 中等 |
| RegMix | 2024 | 回归预测 | 低（<2%训练成本） |
| Mixing Laws | 2024 | 幂律回归建模 | 中等 |
| ADMIRE-BayesOpt | 2025 | 贝叶斯优化 | 低 |
| CLIMB | 2025 | 聚类+迭代自举 | 中等 |
| MixMin | 2025 | 凸最小化 | 低 |
| MergeMix | 2025 | 模型合并 | 低 |

#### 16.1.2 在线方法（动态调整）

最新的趋势是**在训练过程中动态调整数据配比** [^34^]：

- **DBL（Dynamic Batch Loading）**：基于scaling law估计参考最优损失，计算当前超额损失，对学习缓慢的领域增加数据比例
- **DoGE**：通过训练-验证梯度的点积估计"泛化增益"
- **PIKE**：扩展到多任务学习，最小化梯度冲突
- **TiKMix / TANDEM**：最新一代动态混合方法，基于即时梯度反馈更新权重
- **OPUS**：将数据效用与优化器诱导的更新对齐，使用投影评分 [^35^]

### 16.2 合成数据的规模化应用

#### 16.2.1 从补充到核心

合成数据在2024-2025年从实验性补充转变为预训练的核心组成部分 [^33^][^36^]：

- **Nemotron-CC**（NVIDIA, 2024）：从Common Crawl生成了约2万亿合成tokens
- **Phi-4**（Microsoft, 2024）：预训练中使用了**40%合成数据**
- **Qwen-3**（阿里巴巴, 2025）：在训练pipeline中整合了合成数据
- **Kimi K2、Qwen-2.5、Grok、GPT-5**（2025）：都报告了大量使用源重述方法获得的收益

#### 16.2.2 生成范式之争

两种合成数据范式正在竞争 [^33^]：

**生成器驱动**（Generator-driven）：
- 使用大型LLM（如GPT-4）从头生成训练数据
- 优势：质量高、格式可控
- 劣势：成本高、受限于生成器的能力

**源重述**（Source Rephrasing）：
- 使用小型LLM将现有web数据重述为更高质量的格式
- 优势：成本低、覆盖广、多样性高
- 劣势：质量受限于源数据

**2025年趋势**：源重述已成为主导范式 [^33^]。

#### 16.2.3 跨文档合成

最新的WRAP++方法提出了**跨文档合成**（cross-document synthesis），利用web拓扑（如超链接）发现实体关系，将多个文档的事实联合合成为预训练数据。这超越了单文档重述的限制，使模型能够学习**关联性知识** [^37^]。

### 16.3 数据污染与去污染的军备竞赛

#### 16.3.1 污染检测方法

随着LLM越来越强大，数据污染检测成为关键挑战 [^38^]：

**白盒检测**（有模型/数据访问）：
- **N-gram重叠**：LLaMA、PaLM、GPT-4都使用此方法
- **嵌入相似性**：基于余弦相似度的语义匹配
- **层特定检测**（DICE）：分析中间层的激活模式
- **Min-k% Prob**：测试模型是否对某序列赋予异常高的概率

**黑盒检测**（仅有API访问）：
- **时间信息**：利用模型训练窗口和基准发布日期
- **扰动测试**：对基准问题进行改写，观察性能是否下降
- **统计检验**：测试基准的标准顺序是否被"统计上偏爱"

#### 16.3.2 去污染的挑战

去污染面临根本性困难 [^38^]：
1. 精确匹配无法捕获释义、翻译或轻度转换的内容
2. 语义过滤在召回率和精确率之间存在权衡，可能导致过度数据移除
3. 去污染管道通常只在一个时间点应用，无法应对持续预训练、模型蒸馏或合成数据生成带来的间接泄漏
4. 随着训练语料规模和异质性增长，可靠去污染越来越困难和昂贵

#### 16.3.3 新的基准设计理念

为应对污染，研究者提出 [^38^]：
- **动态更新的基准**：定期替换问题
- **私有/受限访问基准**：控制数据传播
- **未解决科学问题基准**：本质上防止模型训练到正确答案
- **数据重构**：转换基准内容以降低其作为训练数据的效用

### 16.4 领域自适应预训练（DAPT）的成熟

#### 16.4.1 DAPT的核心方法

领域自适应预训练（Gururangan et al., 2020）通过在领域特定语料上继续预训练，将通用LLM适配到特定领域 [^39^]：

1. **简单DAPT**：直接在领域语料上继续训练
2. **DAPT+KL**：添加KL散度约束以保持通用性能
3. **DEMix-DAPT**：用领域专家混合层替换FFN层
4. **ELLE**：通过扩展模型宽度/深度高效获取新知识

#### 16.4.2 关键发现

- DAPT显著提升领域任务的**微调**性能，但可能损害**提示**（prompting）性能 [^40^]
- 这表明DAPT在丰富领域知识的同时，可能损害了模型的一般化推理能力
- 灾难性遗忘是DAPT的核心挑战，参数隔离方法可缓解此问题 [^39^]
- DAPT适用于生物医学、金融、法律、科学等专业领域

### 16.5 Scaling Law驱动的数据配比

#### 16.5.1 数据混合律（Data Mixing Laws）

Ye et al.（2024）提出了**数据混合律**，使用幂律回归模型拟合小规模训练运行，预测更大模型的损失 [^34^]。这一方法的核心假设是最优数据配比具有**尺度不变性**（scale invariance）。

然而，AUTOSCALE研究（2024）挑战了这一假设，发现**最优数据组成可能随数据规模变化**——小规模上的最优配比不一定在大规模上保持最优 [^41^]。

#### 16.5.2 从代理模型到大模型的可预测性

最新研究在代理模型方法上取得进展 [^42^]：

- **DataDecide**：构建25个预训练语料库和14个代理规模的受控测试平台，发现连续似然风格指标可以在仅0.01%目标计算量下，对MMLU、ARC、HellaSwag等任务实现**80%+的可预测性**
- **rBridge**：将代理模型的NLL与目标任务的推理轨迹对齐
- **UniMax**：在epoch约束下最大化多样性

### 16.6 2025-2026数据配比的综合趋势

#### 16.6.1 趋势一：从领域配比到细粒度配比

数据配比正在从粗粒度的"领域权重"（如web:code:books = 50:17:8）进化到更细粒度的控制：
- **基于URL的域名级别配比**（RegMix扩展到100个域名）
- **基于质量评分的动态阈值**
- **基于主题的配比**（Topic Over Source）

#### 16.6.2 趋势二：从静态到动态

配比策略从训练前一次性确定，演变为训练过程中的持续调整：
- 早期训练阶段侧重通用web数据
- 中期增加代码和数学数据
- 后期退火阶段使用最高质量数据

#### 16.6.3 趋势三：从人工直觉到自动优化

自动数据配比方法正在取代人工调参：
- 代理模型方法（DoReMi、RegMix、ADMIRE-BayesOpt、CLIMB、MixMin、MergeMix）
- 在线动态调整（DBL、DoGE、TiKMix、OPUS）
- 强化学习驱动（DataAgentRL）

#### 16.6.4 趋势四：数据配比的民主化

RegMix等方法的计算成本极低（<2%训练成本），使得学术界和中小型研究实验室也能进行数据配比研究，不再依赖大型科技公司的计算资源。

[^34^]: "Data Mixture Optimization in Pretraining/Midtraining," survey, 2025-2026.
[^35^]: "OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration," 2025.
[^36^]: Nguyen et al., "BeyondWeb: Lessons from Scaling Synthetic Data," 2025.
[^37^]: "WRAP++: Cross-Document Synthesis for Pretraining," 2025.
[^38^]: "A Survey on Data Contamination for Large Language Models," 2024; "LLM Benchmark Datasets Should Be Contamination-Resistant," 2026.
[^39^]: Gururangan et al., "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks," 2020; "A Survey of Methods for LLM Knowledge Expansion," 2025.
[^40^]: "Adapting Large Language Models to Domains via Reading Comprehension," 2024.
[^41^]: "AUTOSCALE: Automatic Prediction of Compute-Optimal Data Composition for Training LLMs," 2024.
[^42^]: "Forecasting Downstream Performance of LLMs With Proxy Metrics," 2026.

---

## 争议和冲突观点

### 争议一：高质量来源 vs. 大规模web数据

**传统观点**（Gopher、GPT-3时代）：Wikipedia、书籍等"高质量"来源需要过采样，以提升模型质量。

**挑战观点**（Falcon/RefinedWeb）：经过激进去重的纯web数据性能可以超越混合语料 [^5^]。

**调和观点**（LLaMA-3）：web数据是主体（50%），但需要配合专门的代码（17%）、数学/推理（25%）和多语言（8%）数据。

### 争议二：多语言诅咒是否真实？

**传统观点**（XLM-R/BLOOM时代）：增加语言数量必然导致性能下降，尤其是低资源语言 [^19^]。

**新观点**（FineWeb-2, 2025）：诅咒主要来自数据质量和数量不足，而非语言数量本身。高质量过滤后的多语言数据甚至可以带来**多语言收益**而非诅咒 [^31^]。

### 争议三：合成数据的可靠性和规模限制

**乐观观点**：合成数据是解决"数据墙"的终极方案，可以将低质量web数据"回收"为高质量训练数据 [^33^]。

**谨慎观点**：合成数据的质量受限于生成器模型的能力，可能导致"模型崩溃"（model collapse）——模型在合成数据上训练后性能退化。大规模生成高质量合成数据仍缺乏科学理解 [^36^]。

### 争议四：代理模型方法的可扩展性

**支持者**（RegMix, DataDecide）：小规模代理模型足以预测大规模训练的最优配比，rank invariance假设基本成立 [^25^][^42^]。

**质疑者**（AUTOSCALE）：最优数据组成随数据规模变化，小规模上的最优配比不一定在大规模上最优 [^41^]。对于数学和代码等复杂能力，小代理实验往往不够准确。

---

## 建议深入研究的领域

### 高优先级

1. **数据混合的缩放律**：建立从代理模型到目标模型的可靠映射理论，理解在何种条件下rank invariance成立
2. **动态数据配比的收敛性**：证明在线数据调整方法（如DBL、OPUS）的收敛保证
3. **合成数据质量的理论分析**：理解合成数据何时有效、何时导致模型崩溃
4. **多语言数据配比的最优策略**：探索打破"多语言诅咒"的系统方法

### 中优先级

5. **代码数据提升推理的机理**：深入理解代码数据为什么、在何种条件下提升推理能力
6. **数据污染的完整解决方案**：开发可扩展的污染检测和去污染pipeline
7. **领域自适应的遗忘问题**：理解并解决DAPT中的灾难性遗忘
8. **数据配比的公平性**：确保数据选择不会放大社会偏见

### 方法论工具

9. **DataComp-LM的持续扩展**：增加更多评估任务、更大模型规模、更多模态
10. **开源数据配比工具链**：开发标准化的数据配比优化框架

---

## 重要论文和来源索引

### 核心论文

| 论文 | 作者/机构 | 年份 | 主题 |
|---|---|---|---|
| The Pile | Gao et al., EleutherAI | 2020 | 多样预训练语料 |
| Scaling Language Models: Methods, Analysis & Insights from Training Gopher | Rae et al., DeepMind | 2021 | 规模化训练分析 |
| Training Compute-Optimal Large Language Models | Hoffmann et al., DeepMind | 2022 | Chinchilla缩放律 |
| DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | Xie et al., Google | 2023 | 数据配比优化 |
| Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al., Meta | 2023 | LLaMA系列 |
| The Llama 3 Herd of Models | Meta | 2024 | LLaMA-3/3.1 |
| DataComp-LM: In search of the next generation of training sets | Li et al., Apple/AI2 | 2024 | 数据筛选基准 |
| RegMix: Data Mixture as Regression | Liu et al. | 2024 | 回归优化配比 |
| MATES: Model-Aware Data Selection | Yu et al., CMU | 2024 | 动态数据选择 |
| Nemotron-CC | Su et al., NVIDIA | 2024 | 合成数据 |
| WRAP: Web Rephrase Augmented Pre-training | Maini et al. | 2024 | 源重述方法 |
| The FineWeb Datasets | Penedo et al., HuggingFace | 2024 | 高质量web语料 |
| BeyondWeb: Lessons from Scaling Synthetic Data | Nguyen et al. | 2025 | 合成数据缩放 |
| Revisiting Multilingual Data Mixtures | 2025 | 多语言诅咒再审视 |
| OPUS: Data Selection in Every Iteration | 2025 | 在线动态选择 |

### 关键来源

- **EleutherAI**：The Pile、Pythia
- **DeepMind**：Gopher、Chinchilla
- **Meta**：LLaMA系列、Code Llama
- **Hugging Face**：FineWeb、FineWeb-Edu、Cosmopedia
- **Google**：DoReMi、T5/C4、PaLM
- **Apple/AI2/Facebook**：DataComp-LM
- **NVIDIA**：Nemotron系列
- **BigCode/ServiceNow**：StarCoder系列
- **BigScience**：BLOOM/ROOTS
- **TII**：Falcon/RefinedWeb

---

*本研究素材基于2024-2025年的最新学术论文、技术博客和官方文档，为《大模型预训练技术演进》书籍第13-16章提供素材支撑。*
