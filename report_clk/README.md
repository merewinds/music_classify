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

论文引用的图件已同步到本目录的 `figures/`。
