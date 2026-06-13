# Version-3 final experiment

The reproducible experiment used by `report_clk` is:

```powershell
python code\final_experiment.py
```

For a fast end-to-end verification:

```powershell
python code\final_experiment.py --quick
```

Useful options are `--per-genre`, `--points`, `--workers`, and
`--force-recompute`. The default target is 100 pieces per genre. If one genre
has fewer usable duplicate-free groups, the sampler automatically uses the
largest common balanced size.

The v3 pipeline is split by responsibility:

- `data_pipeline.py`: title, SHA-256, and normalized-melody duplicate grouping;
- `distance_models.py`: Hausdorff family, phase-aligned RMSE, and multivariate DTW;
- `evaluation.py`: nested grouped validation, confidence intervals, fusion,
  ablation, and corrected model comparisons;
- `provenance.py`: environment, data, command, metric, and artifact ledger;
- `final_experiment.py`: orchestration and the stable output contract.

MIDI note-on parsing is implemented locally in `midi_geometry.py`; no external
MIDI library is required. Python dependencies are listed in `requirements.txt`.

The experiment writes:

- cached balanced samples and distance matrices to `results/final/cache/`;
- leakage and dataset audit tables to `results/final/tables/`;
- all outer-fold predictions to `results/final/predictions_primary.npz`;
- a machine-readable result summary to `results/final/summary.json`;
- a provenance ledger to `results/final/run_manifest.json`;
- synchronized LaTeX values to `report_clk/generated_results.tex`.

After the numerical experiment is complete, generate the final Chinese paper
assets with:

```powershell
python code\paper_assets.py
```

This reuses the cached folds and distance matrices, then writes:

- presentation CSV files to `results/final/tables_paper/`;
- unified 300 dpi Chinese figures to `results/final/figures_paper/`;
- synchronized report figures to `report_clk/figures/`.

Compile the paper with:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd report_clk\main.tex
```

The older scripts in this directory are exploratory baselines only. They use
earlier preprocessing or holdout protocols and are not sources for final-paper
claims.
