# cof-condensate-bursting

[![DOI](https://zenodo.org/badge/1286603385.svg)](https://doi.org/10.5281/zenodo.21689854)

Code and data to accompany the manuscript titled **"Condensates increase the frequency of transcription"**
on CoF condensate-driven transcriptional bursting. Simulates a stochastic three-state 
promoter model, comparing a **soluble cofactor** regime against a 
**condensate-forming cocofactor** regime.

The `Code` folder contains the Gillespie simulation engine. A `tutorial_3state.ipynb`
notebook is provided to get familiarized with the model.

The `Figures` folder contains the data and notebook used to produce the figures in
the manuscript. Simulation output lives in `figure-data/`; generated panels are
written to `Output/`.

Environment file `cof-condensate-bursting-env.yml` includes all of the code
dependencies.

## Quick start

```bash
conda env create -f cof-condensate-bursting-env.yml
conda activate cof-condensate-bursting
cd Figures
jupyter lab generate_figures.ipynb
```

Figures are written to `Figures/Output/<date>/`

All simulation parameters used to produce these figures are described in the
Methods section. This repo provides the model and the code to run and explore it.
