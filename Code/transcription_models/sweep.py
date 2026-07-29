"""
Population Sweep Driver
========================

Assembles per-cell heterogeneity (heterogeneity.py), the binding/condensate rate laws
(rate_calculators.py), and the Gillespie simulation (gillespie.py) into the CoF
dose-response sweeps used throughout the paper: for each nominal CoF concentration,
simulate a population of cells, each with its own jittered CoF and sampled rate
parameters, and record per-cell burst statistics.

`run_sweep` runs one model (binding, or condensate at a given threshold) across the
full CoF grid. `run_full_sweep` runs the binding model plus every condensate threshold
in `cfg['threshold_values']` and assembles the dict that gets pickled to
`figure-data/bursty_transcription_data.pkl`.

`cfg` throughout is a dict shaped like the one built in `Code/01_tutorial.ipynb`'s
Part 4.
"""

import numpy as np
from tqdm import tqdm

from .analysis import analyze_bursts
from .gillespie import ThreeStateModel, gillespie_simulation, gillespie_simulation_windowed
from .heterogeneity import (
    jitter_concentrations,
    sample_condensate_localization,
    sample_rate_parameters,
    sample_threshold,
)
from .rate_calculators import calculate_rates_binding, calculate_rates_condensate


def run_sweep(cfg, model_type, CoF_threshold_nom, thresh_idx=0, show_progress=True):
    """Sweep CoF with per-cell jittered CoF driving physics.

    Parameters
    ----------
    cfg : dict
        Loaded `config/simulation_params.yaml`.
    model_type : {'binding', 'condensate'}
        Which rate law to use.
    CoF_threshold_nom : float
        Nominal condensation threshold [CoF*] (ignored for `model_type='binding'`).
    thresh_idx : int
        Index of this threshold among `cfg['threshold_values']`, used only to keep
        random seeds distinct across thresholds.
    show_progress : bool
        Whether to display a tqdm progress bar over the CoF grid.

    Returns
    -------
    dict with one entry per per-cell metric (`transcripts`, `n_bursts`, `burst_freq`,
    `burst_duration`, `burst_size`, `localization_windows`, `sampled_thresholds`,
    `sampled_params`, `x_jittered`), each keyed `{0: {grid_index: [per-cell values]}}`.
    """
    rates_cfg = cfg['rates']
    het = cfg['heterogeneity']
    sim = cfg['simulation']

    tf = cfg['tf_fixed']
    cof_grid = cfg['cof_grid']
    CoF_concs = np.logspace(cof_grid['log10_min'], cof_grid['log10_max'], cof_grid['n_points'])
    n_traj = sim['n_trajectories']
    t_max = sim['t_max']
    window_size = sim['window_size']
    base_seed = sim['base_seed']
    kt_factor = rates_cfg['kt_cof_factor']
    jitter_cv = het['jitter_cv']
    P_gene = het['p_gene_localization']
    dense_phase = cfg['dense_phase_cof']
    off_gene_ref = cfg['off_gene_dilute_conc']

    k2_bind = rates_cfg['k2_base_binding']
    k3_bind = rates_cfg['k3_base_binding']
    k2_cond = k2_bind
    k3_low = k3_bind
    k3_max = k3_low * dense_phase
    pr_max = rates_cfg['pr_max']
    koff_2 = rates_cfg['koff_2']
    koff_3 = rates_cfg['koff_3']
    kt_base = rates_cfg['kt_base']

    n_CoF = len(CoF_concs)
    transcripts = {0: {}}
    n_bursts_dict = {0: {}}
    burst_freq_dict = {0: {}}
    burst_duration_dict = {0: {}}
    burst_size_dict = {0: {}}
    localization_windows = {0: {}}
    sampled_thresholds = {0: {}}
    sampled_params = {0: {}}
    x_jittered = {0: {}}

    pbar = tqdm(total=n_CoF, desc=f'{model_type} thresh={CoF_threshold_nom:.2f}', disable=not show_progress)

    for j, CoF_nom in enumerate(CoF_concs):
        traj_transcripts, traj_n_bursts = [], []
        traj_burst_freq, traj_burst_dur, traj_burst_size = [], [], []
        traj_windows, traj_thresholds, traj_params_list = [], [], []

        jitter_seed = base_seed + 4_000_000 + j
        x_jit = jitter_concentrations(CoF_nom, n_traj, jitter_CV=jitter_cv, seed=jitter_seed)
        x_jittered[0][j] = x_jit

        for k in range(n_traj):
            cell_seed = base_seed + thresh_idx * 10_000_000 + j * n_traj + k
            cell_CoF = float(x_jit[k])

            if model_type == 'binding':
                rates = sample_rate_parameters(
                    k2_base=k2_bind, k3_low=k3_bind, k3_max=k3_bind,
                    kt_base=kt_base, PR_max=1.0, koff_2=koff_2, koff_3=koff_3,
                    k2_CV=het['k2_cv'], k3_low_CV=het['k3_low_cv'], k3_max_CV=0.0,
                    kt_CV=het['kt_cv'], PR_CV=0.0,
                    koff_2_CV=het['koff_2_cv'], koff_3_CV=het['koff_3_cv'],
                    seed=cell_seed,
                )
            else:
                rates = sample_rate_parameters(
                    k2_base=k2_cond, k3_low=k3_low, k3_max=k3_max,
                    kt_base=kt_base, PR_max=pr_max, koff_2=koff_2, koff_3=koff_3,
                    k2_CV=het['k2_cv'], k3_low_CV=het['k3_low_cv'], k3_max_CV=het['k3_max_cv'],
                    kt_CV=het['kt_cv'], PR_CV=het['pr_cv'],
                    koff_2_CV=het['koff_2_cv'], koff_3_CV=het['koff_3_cv'],
                    seed=cell_seed,
                )
            traj_params_list.append(rates)

            if model_type == 'condensate':
                cell_threshold = sample_threshold(
                    CoF_threshold_nom, CV=het['cof_threshold_cv'],
                    seed=cell_seed + 1_000_000,
                )
            else:
                cell_threshold = CoF_threshold_nom
            traj_thresholds.append(cell_threshold)

            if model_type == 'binding':
                k2, k3, kt = calculate_rates_binding(
                    tf, cell_CoF,
                    k2_base=rates['k2_base'], k3_base=rates['k3_low'],
                    koff_2=rates['koff_2'], koff_3=rates['koff_3'],
                    kt_base=rates['kt_base'], kt_CoF_factor=kt_factor,
                )
                model = ThreeStateModel(
                    k2=k2, koff_2=rates['koff_2'], k3=k3, koff_3_to_2=rates['koff_3'],
                    kt=kt, allow_3_to_1=False, allow_3_to_2=True, instant_transcription=False,
                )
                result = gillespie_simulation(model, t_max, seed=cell_seed + 3_000_000)
                traj_windows.append([True])

            elif window_size >= t_max:
                at_gene = sample_condensate_localization(
                    P_gene, cell_CoF, cell_threshold, seed=cell_seed + 2_000_000,
                )
                k2, k3, kt = calculate_rates_condensate(
                    tf, cell_CoF,
                    k2_base=rates['k2_base'], k3_low=rates['k3_low'], k3_max=rates['k3_max'],
                    CoF_threshold=cell_threshold, PR_max=rates['PR_max'],
                    koff_2=rates['koff_2'], koff_3=rates['koff_3'],
                    kt_base=rates['kt_base'], kt_CoF_factor=kt_factor,
                    dense_phase_CoF=dense_phase, P_at_gene=at_gene,
                    off_gene_dilute_conc=off_gene_ref,
                )
                model = ThreeStateModel(
                    k2=k2, koff_2=rates['koff_2'], k3=k3, koff_3_to_2=rates['koff_3'],
                    kt=kt, allow_3_to_1=False, allow_3_to_2=True, instant_transcription=False,
                )
                result = gillespie_simulation(model, t_max, seed=cell_seed + 3_000_000)
                traj_windows.append([at_gene])

            else:
                def model_factory(at_gene, seed, _CoF=cell_CoF, _rates=rates, _thr=cell_threshold):
                    k2, k3, kt = calculate_rates_condensate(
                        tf, _CoF,
                        k2_base=_rates['k2_base'], k3_low=_rates['k3_low'], k3_max=_rates['k3_max'],
                        CoF_threshold=_thr, PR_max=_rates['PR_max'],
                        koff_2=_rates['koff_2'], koff_3=_rates['koff_3'],
                        kt_base=_rates['kt_base'], kt_CoF_factor=kt_factor,
                        dense_phase_CoF=dense_phase, P_at_gene=at_gene,
                        off_gene_dilute_conc=off_gene_ref,
                    )
                    return ThreeStateModel(
                        k2=k2, koff_2=_rates['koff_2'], k3=k3, koff_3_to_2=_rates['koff_3'],
                        kt=kt, allow_3_to_1=False, allow_3_to_2=True, instant_transcription=False,
                    )

                def localization_sampler(seed, _CoF=cell_CoF, _thr=cell_threshold):
                    return sample_condensate_localization(P_gene, _CoF, _thr, seed=seed)

                result, window_states = gillespie_simulation_windowed(
                    model_factory=model_factory, t_max=t_max, window_size=window_size,
                    localization_params={
                        'P_gene_localization': P_gene,
                        'CoF_conc': cell_CoF,
                        'CoF_threshold': cell_threshold,
                        'localization_sampler': localization_sampler,
                    },
                    initial_state=0, seed=cell_seed + 3_000_000,
                )
                traj_windows.append(window_states)

            traj_transcripts.append(int(result.transcript_counts[-1]))
            burst_stats = analyze_bursts(result, active_state=2, min_burst_transcripts=1)
            nb = burst_stats['n_bursts']
            traj_n_bursts.append(nb)
            if nb > 0:
                traj_burst_freq.append(burst_stats['burst_frequency'])
                traj_burst_dur.append(burst_stats['mean_burst_duration'])
                traj_burst_size.append(burst_stats['mean_burst_size'])
            else:
                traj_burst_freq.append(0.0)
                traj_burst_dur.append(np.nan)
                traj_burst_size.append(np.nan)

        transcripts[0][j] = traj_transcripts
        n_bursts_dict[0][j] = traj_n_bursts
        burst_freq_dict[0][j] = traj_burst_freq
        burst_duration_dict[0][j] = traj_burst_dur
        burst_size_dict[0][j] = traj_burst_size
        localization_windows[0][j] = traj_windows
        sampled_thresholds[0][j] = traj_thresholds
        sampled_params[0][j] = traj_params_list
        pbar.update(1)

    pbar.close()
    return {
        'transcripts': transcripts,
        'n_bursts': n_bursts_dict,
        'burst_freq': burst_freq_dict,
        'burst_duration': burst_duration_dict,
        'burst_size': burst_size_dict,
        'localization_windows': localization_windows,
        'sampled_thresholds': sampled_thresholds,
        'sampled_params': sampled_params,
        'x_jittered': x_jittered,
    }


def run_full_sweep(cfg, show_progress=True):
    """Run the binding model plus every condensate threshold in `cfg['threshold_values']`.

    Returns the dict that `run_simulations.py` pickles to `cfg['output']['pkl_path']`.
    """
    cof_grid = cfg['cof_grid']
    CoF_concentrations = np.logspace(cof_grid['log10_min'], cof_grid['log10_max'], cof_grid['n_points'])
    het = cfg['heterogeneity']
    sim = cfg['simulation']
    off_gene_ref = cfg['off_gene_dilute_conc']

    results_binding = run_sweep(cfg, 'binding', CoF_threshold_nom=1.0, show_progress=show_progress)

    results_condensate = {}
    for thresh_idx, thresh in enumerate(cfg['threshold_values']):
        results_condensate[thresh] = run_sweep(
            cfg, 'condensate', thresh, thresh_idx=thresh_idx, show_progress=show_progress,
        )

    return {
        'TF_fixed': cfg['tf_fixed'],
        'CoF_concentrations': CoF_concentrations,
        'threshold_values': cfg['threshold_values'],
        'kt_CoF_factor': cfg['rates']['kt_cof_factor'],
        'n_trajectories': sim['n_trajectories'],
        't_max': sim['t_max'],
        'window_size': sim['window_size'],
        'CoF_threshold_CV': het['cof_threshold_cv'],
        'P_gene_localization': het['p_gene_localization'],
        'jitter_CV': het['jitter_cv'],
        'jitter_is_true_CoF': True,
        'fixed_threshold': True,
        'off_gene_dilute_conc': off_gene_ref,
        'dense_phase_CoF': cfg['dense_phase_cof'],
        'results_binding': results_binding,
        'results_condensate_by_threshold': results_condensate,
    }
