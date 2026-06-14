# LaTeX 论文终稿

本目录保存全国大学生数学建模竞赛论文风格的匿名终稿。正文入口为
`main.tex`，已编译的交付文件为 `main.pdf`。

在仓库根目录安装依赖并运行完整实验：

```powershell
python -m pip install -r requirements.txt
python code\final_experiment.py --per-genre 100 --points 96 `
  --sample-seeds 2026 2027 2028 `
  --outer-seeds 11 23 42 67 101 `
  --primary-only-repeats
python code\paper_assets.py
```

只做快速端到端检查时，可以运行：

```powershell
python code\final_experiment.py --quick
python -m unittest discover -s code -p test_pipeline.py -v
```

使用 XeLaTeX 编译论文：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd report_clk\main.tex
```

实验流水线将论文核心数值写入 `generated_results.tex`，并将最终中文图表
同步到 `figures/`。因此克隆仓库后无需下载原始 MIDI 数据或保留
`results/` 目录，也可以直接编译现有论文。只有重新运行数值实验时才需要
准备 `adl-piano-midi/` 数据集。

最终结论来自分组嵌套交叉验证、三个样本种子与五个外层种子的完整重复、
条件 Bootstrap 区间、完整运行配对差异和重复特征消融。`results/final/`
中的缓存、审计明细和实验清单均为
可再生产物，不纳入 Git；论文实际引用的 PNG、自动生成 LaTeX 宏与最终
PDF 则保存在本目录并纳入版本控制。
