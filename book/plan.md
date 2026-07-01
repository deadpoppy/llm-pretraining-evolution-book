# 书籍撰写计划：《大语言模型预训练技术演进》

## 阶段一：深度研究（Stage 1: Deep Research）
- **技能**: `deep-research-swarm`
- **目标**: 对30章涉及的技术主题进行全面研究，收集关键论文、技术细节、时间线、对比数据
- **子任务**:
  - Research Team A: 预训练前史至Transformer（第1-2章）
  - Research Team B: BERT/GPT/T5路线分化（第3-5章）
  - Research Team C: Tokenizer演进（第6-8章）
  - Research Team D: Scaling Law（第9-12章）
  - Research Team E: 数据配比演进（第13-16章）
  - Research Team F: 分布式训练（第17-21章）
  - Research Team G: 架构与目标函数新变化（第22-25章）
  - Research Team H: 2024-2026最新技术专题（第26-30章）
- **输出**: 研究简报文件（research briefs），每章提供关键事实、论文引用、技术对比表格素材

## 阶段二：大纲精化（Stage 2: Outline Design）
- **技能**: `report-writing` (Phase 1)
- **目标**: 基于研究结果，将用户提供的目录精化为每章详细大纲（含节标题、每节要点、表格/图表规划）
- **输出**: `outline.md`

## 阶段三：内容撰写（Stage 3: Content Writing）
- **技能**: `report-writing` (Phase 2)
- **目标**: 按批次并行撰写各章节
- **约束**: 内容充实、言简意赅、优先表格/管线图、无代码、无废话
- **批次**:
  - Batch 1: 导读 + 第1-5章（预训练前史到路线分化）
  - Batch 2: 第6-8章（Tokenizer）
  - Batch 3: 第9-12章（Scaling Law）
  - Batch 4: 第13-16章（数据配比）
  - Batch 5: 第17-21章（分布式训练）
  - Batch 6: 第22-25章（架构与目标函数新变化）
  - Batch 7: 第26-30章 + 结语（最新技术专题）
- **输出**: 各章 Markdown 文件

## 阶段四：审校与整合（Stage 4: Review & Assembly）
- **技能**: `report-writing` (Phase 3)
- **目标**: 审校所有章节，检查一致性、准确性、风格，整合为完整书籍
- **输出**: `final.md`

## 阶段五：格式转换（Stage 5: DOCX Production）
- **技能**: `docx`
- **目标**: 将最终 Markdown 转换为 Word 文档
- **输出**: `大语言模型预训练技术演进.docx`
