<section class="book-hero" markdown>

# 大语言模型预训练技术演进

<p class="book-subtitle">从 Scaling Law 到数据配方与分布式训练</p>

<div class="book-actions" markdown>
[开始阅读](chapters/llm_pretraining_sec00.md){ .md-button .md-button--primary } [查看完整书稿](full-book.md){ .md-button } [GitHub 仓库](https://github.com/deadpoppy/llm-pretraining-evolution-book){ .md-button }
</div>

</section>

<div class="book-meta" markdown>
<div markdown>
**主题**

大语言模型预训练的技术史、方法论和工程脉络。
</div>

<div markdown>
**时间线**

2013-2026，从词向量到推理模型和新一代训练系统。
</div>

<div markdown>
**形式**

32 个分章页面，支持搜索、目录、公式、图表和脚注。
</div>
</div>

## 这本书讲什么

本书梳理大语言模型预训练技术如何从早期词向量、RNN/LSTM 和 Transformer，演进到 GPT/BERT/T5、Scaling Law、Chinchilla、数据配方、分布式训练、MoE、长上下文、多模态预训练、LLaMA 3、DeepSeek-V3 和推理模型。

它不是代码教程，而是一本面向技术判断的书：帮助读者理解为什么某条路线成为主流，关键实验改变了什么，数据、模型、算力和系统工程之间如何相互制约。

## 推荐阅读路径

**速读路线**：导读 → 第2章 Transformer → 第3章 GPT 路线 → 第9-11章 Scaling Law 与 Chinchilla → 第26-27章 LLaMA 3 与 DeepSeek-V3。

**精读路线**：按左侧目录从第0章读到结语。Tokenizer、Scaling Law、数据配方和分布式训练几条主线分别贯穿多个章节，适合连着读。

## 主要内容

- 预训练前史：Word2Vec、GloVe、RNN/LSTM、Seq2Seq 与 Attention。
- Transformer 与三条路线：GPT、BERT、T5/BART。
- Scaling Law 与 Chinchilla：参数量、数据量和计算量的配比。
- 数据工程：Common Crawl、The Pile、FineWeb、代码/数学/多语言数据。
- Tokenizer 与目标函数：BPE、WordPiece、NTP、MLM、MTP。
- 训练系统：数据并行、张量并行、流水线并行、ZeRO、FlashAttention、FP8。
- 代表案例：LLaMA 3、DeepSeek-V3、Qwen3 与新一代开源模型。
- 未来趋势：合成数据、推理模型、持续预训练、多模态和数据枯竭。

## 渲染能力

这个站点使用 MkDocs Material 构建，开启了：

- 左侧章节导航、页内目录、全文搜索和深色模式。
- Markdown 表格、脚注、定义列表、任务列表、代码高亮和复制按钮。
- Mermaid 图表渲染。
- MathJax 公式渲染，支持 `$...$`、`$$...$$`、`\(...\)`、`\[...\]`。
