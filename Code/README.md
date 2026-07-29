# Code

The stochastic three-state promoter simulation engine. A `tutorial_3state.ipynb` notebook
is provided to get familiarized with the model.

The main simulation logic is in the `transcription_models` package: `gillespie.py`
runs the Gillespie algorithm, `rate_calculators.py` computes binding vs condensate
rate laws, `heterogeneity.py` samples per-cell variability (jittered concentrations,
rate parameters, condensate localization), `sweep.py` assembles those into the
population sweeps (`run_sweep`), and `analysis.py`
extracts burst statistics from a simulated trajectory.

`tutorial_3state.ipynb` walks through the model one piece at a time — a single simulated
cell, the binding vs. condensate rate laws, cell-to-cell heterogeneity — and finishes
by calling `run_sweep` directly, at a much smaller scale than the paper's full run.
