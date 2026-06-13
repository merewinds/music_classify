# LaTeX 终稿

Compile with XeLaTeX from this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode main.tex
```

`main.pdf` 采用全国大学生数学建模竞赛论文的正文风格：第一页直接为题目、
摘要和关键词，不设目录和课程式大封面。学生信息为崔立坤，完成时间为
2026 年 6 月。

所有数值结论来自 `../results/final/summary.json` 与
`../results/final/tables/`。中文终稿图和展示型 CSV 由
`../code/paper_assets.py` 生成，分别位于：

- `../results/final/figures_paper/`
- `../results/final/tables_paper/`

论文引用的图件已同步到本目录的 `figures/`，因此直接编译 `main.tex` 不需要
`results/` 目录。只有重跑完整实验或重新生成论文图表与展示型 CSV 时，才需要
先生成 `results/final/` 下的对应文件。

当前实验缓存版本为 `v2`。复现前可先运行基础回归测试：

```powershell
D:\app\Anaconda\python.exe -m unittest code.test_pipeline -v
```

最终流程采用每曲名组一个版本、组级标签置换、重复内层三折调参，以及五组外层
随机分折稳定性验证。`../results/final/tables_paper/` 另含两种模型的混淆矩阵和
随机森林折外置换重要性，便于逐项核查论文结论。
