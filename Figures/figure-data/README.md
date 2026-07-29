# figure-data

`bursty_transcription_data.pkl` is the paper's simulation output. It is a single dict:

```python
{
    'TF_fixed', 'CoF_concentrations', 'threshold_values',
    'n_trajectories', 't_max', 'jitter_CV', 'jitter_is_true_CoF',
    'fixed_threshold', 'dense_phase_CoF', 'off_gene_dilute_conc', ...
    'results_binding': {transcripts, burst_freq, burst_size, x_jittered, ...},
    'results_condensate_by_threshold': {0.25: {...}, 0.5: {...}, 1.0: {...}, ...},
}
```

`CoF_concentrations` is a 20-point log-spaced grid (0.1 to 10). Each `results_*` entry
holds per-cell metrics (500 cells per grid point) for the binding model and, per
condensate threshold, the condensate model — including `x_jittered`, each cell's
per-cell jittered CoF, which drives both the simulated rates and the x-axis in scatter
plots.

`illustrative_burst_traces.npz` holds two arrays, `bursts_no` and `bursts_yes`, each an
array of `(start, end, height)` rows describing manually transcribed illustrative
trajectories used only for Figure 1 Panel I. These are hand-chosen trajectories to schematically illustrate the two
regimes (short/infrequent vs. long/frequent transcriptional activity).

`rebinned_summary.pkl` is a sliding-window summary of `bursty_transcription_data.pkl`,
loaded directly by `../generate_figures.ipynb`. It is a single dict:

```python
{
    'x_flat': {'binding': {...}, 'condensate': {...}},
    'y_flat': {'binding': {...}, 'condensate': {...}},
    'rebin_narrow': {'transcripts': {...}, 'burst_freq': {...}, 'burst_size': {...}},
    'rebin_wide': {'transcripts': {...}, 'burst_freq': {...}, 'burst_size': {...}},
}
```

`x_flat`/`y_flat` hold each cell's jittered CoF and metric value, flattened across the
CoF grid, per model and metric — the inputs to the single-cell scatter panels.
`rebin_narrow`/`rebin_wide` hold, per metric, the sliding-window mean/std of that
metric against CoF for the binding and condensate models, over the narrow `[0.5, 2.0]`
and wide `[0.1, 10.0]` CoF ranges respectively — the inputs to the line panels.

`processed/` holds the sliding-window mean and std tables that feed each line panel —
`transcripts_{narrow,wide}_means.csv`, `freq_active_state_{narrow,wide}_means.csv`, and
`mean_transcripts_active_{narrow,wide}_means.csv` — for readers who want the numbers
without running Python.
