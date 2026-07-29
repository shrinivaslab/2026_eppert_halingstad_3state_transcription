# Figures

Data and notebook for producing the figures in the manuscript. The raw simulation
data and its precomputed summary live in `figure-data/`; generated panels are written
to `Output/`.

Figures are produced entirely within the `generate_figures.ipynb` notebook, which
loads `figure-data/rebinned_summary.pkl` and `figure-data/illustrative_burst_traces.npz`,
defines its own plotting helpers (styling, panel functions) inline, builds and
displays each panel, and writes PNG/PDF files to a dated subfolder of `Output/`.

`figure-data/` holds `bursty_transcription_data.pkl` (the paper's raw simulation
output), `rebinned_summary.pkl` (its precomputed sliding-window summary, loaded
directly by the notebook), and `illustrative_burst_traces.npz` (specific example
trajectories from simulation used only for Figure 1 Panel I).

`Output/<date>/` is the bundled record of the figures —
regenerating the figures writes to a new dated folder rather than overwriting it.

`figure-data/processed/*.csv` holds the sliding-window mean/std tables underlying
each line panel.
