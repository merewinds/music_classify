# Final experiment entry point

The reproducible experiment used by `report_clk` is:

```powershell
D:\app\Anaconda\python.exe code\final_experiment.py
```

It uses only NumPy, SciPy, scikit-learn, and Matplotlib.  MIDI note-on parsing is
implemented locally in `midi_geometry.py`, so `music21`, `pretty_midi`, and
`dtaidistance` are not required.

The script writes:

- cached balanced samples and distance matrices to `results/final/cache/`;
- numerical tables to `results/final/tables/`;
- paper-ready figures to `results/final/figures/`;
- a machine-readable summary to `results/final/summary.json`.

After the numerical experiment is complete, generate the final Chinese paper
assets with:

```powershell
D:\app\Anaconda\python.exe code\paper_assets.py
```

This reuses the cached folds and distance matrices, then writes:

- presentation CSV files to `results/final/tables_paper/`;
- unified 300 dpi Chinese figures to `results/final/figures_paper/`;
- synchronized report figures to `report_clk/figures/`.

The older scripts in this directory are retained as exploratory baselines.  They
are not the source of the final report's numerical claims.
