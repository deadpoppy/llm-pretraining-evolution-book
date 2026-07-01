# 大语言模型预训练技术演进

> 从 Scaling Law 到数据配方与分布式训练

这是一本梳理大语言模型预训练技术演进的中文技术书。全书围绕 2013-2026 年大模型预训练的关键转折展开，重点解释 Scaling Law、数据配方、Tokenizer、预训练目标函数、分布式训练系统、MoE、长上下文、开源模型训练实践和推理模型对预训练范式的反向影响。

本书不是代码教程，而是一本偏技术史、方法论和工程脉络的长文档。它适合希望系统理解大模型预训练如何从词向量、Transformer、GPT/BERT/T5，走到 Chinchilla、LLaMA、DeepSeek、MoE、长上下文和 2026 年后训练趋势的读者。

## 在线地址

- 网站地址：https://deadpoppy.github.io/llm-pretraining-evolution-book/
- GitHub 仓库：https://github.com/deadpoppy/llm-pretraining-evolution-book

## 阅读入口

- 在线书站：[https://deadpoppy.github.io/llm-pretraining-evolution-book/](https://deadpoppy.github.io/llm-pretraining-evolution-book/)
- 完整书稿：[book/大语言模型预训练技术演进.md](book/大语言模型预训练技术演进.md)
- 分章稿件：[book/chapters/](book/chapters/)
- 站点源码：[docs/](docs/)
- 插图资源：[book/assets/](book/assets/)
- 深度研究材料：[book/research/](book/research/)
- Word 版本：[book/docx/](book/docx/)

## 内容主线

全书按技术演进脉络组织，主要覆盖：

1. 预训练前史：Word2Vec、GloVe、RNN/LSTM、Seq2Seq 与 Attention。
2. Transformer 的结构突破：Self-Attention、Encoder/Decoder 架构与规模化工程基础。
3. GPT、BERT、T5/BART 三条预训练路线的形成与分化。
4. Scaling Law 与 Chinchilla：参数量、数据量、计算量之间的最优配比。
5. 数据工程：Common Crawl、The Pile、RefinedWeb、FineWeb、代码/数学/多语言数据配方。
6. Tokenizer 与目标函数：BPE、WordPiece、字节级 BPE、MLM、NTP、MTP。
7. 训练系统：数据并行、张量并行、流水线并行、ZeRO、FlashAttention、FP8。
8. 架构与实践案例：MoE、长上下文、LLaMA 3、DeepSeek-V3、开源模型追赶。
9. 未来趋势：合成数据、推理模型、持续预训练、多模态与数据枯竭问题。

## 仓库结构

```text
.
├── README.md
├── mkdocs.yml
├── requirements.txt
├── .github/workflows/pages.yml
├── book/
│   ├── 大语言模型预训练技术演进.md
│   ├── outline.md
│   ├── plan.md
│   ├── assets/
│   ├── chapters/
│   ├── docx/
│   └── research/
└── docs/
    ├── index.md
    ├── full-book.md
    ├── assets/
    ├── chapters/
    ├── research/
    ├── javascripts/
    └── stylesheets/
```

## 说明

仓库保留了完整合并稿、逐章 Markdown、插图、研究材料和 Word 文档，方便在线阅读、二次编辑和版本管理。网站使用 MkDocs Material 构建，已开启章节导航、全文搜索、代码高亮、表格、脚注、Mermaid 图和 MathJax 公式渲染。

## 本地预览

```bash
pip install -r requirements.txt
mkdocs serve
```
