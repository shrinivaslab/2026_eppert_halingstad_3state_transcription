"""
Cell-to-Cell Heterogeneity Module
=================================

Functions for modeling cell-to-cell variability in transcriptional bursting
parameters. Biological cells, even genetically identical ones, exhibit
substantial heterogeneity due to differences in:

- Nuclear microenvironment (salt, crowding, PTMs) -> threshold variability
- Chromatin state and enhancer activity -> rate variability
- Stochastic condensate positioning -> localization variability

All sampling uses log-normal distributions, which are strictly positive
and appropriate for biochemical rate constants.

When all CV parameters are 0 and P_gene_localization is 1.0, behavior
is identical to the deterministic (non-heterogeneous) model.
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ============================================================================
# Default CV values
# ============================================================================

# Threshold heterogeneity
COF_THRESHOLD_CV_DEFAULT = 0.25  # 25% coefficient of variation

# Rate heterogeneity - on-rates have more variability (expression-level driven)
K2_BASE_CV_DEFAULT = 0.3
K3_LOW_CV_DEFAULT = 0.3
K3_MAX_CV_DEFAULT = 0.3
KT_BASE_CV_DEFAULT = 0.25
PR_MAX_CV_DEFAULT = 0.2

# Off-rates: NO variability by default (driven by molecular chemistry)
KOFF_2_CV_DEFAULT = 0.0
KOFF_3_CV_DEFAULT = 0.0

# Condensate localization
P_GENE_LOCALIZATION_DEFAULT = 0.85  # 85% of condensates form at gene
SQUELCHING_FACTOR_DEFAULT = None    # No squelching by default


def _sample_normal(mean: float, cv: float, rng: np.random.RandomState) -> float:
    """
    Sample from a normal (Gaussian) distribution with specified mean and CV.

    The normal distribution is parameterized so that:
        E[X] = mean
        CV(X) = std(X) / E[X] = cv

    NOTE: The normal distribution has support on all of (-inf, +inf), so it
    can in principle return negative values. Since rate constants must be
    strictly positive, the sample is clamped to a small positive floor.
    For large CV values (>0.5 or so), consider using _sample_lognormal instead,
    which is strictly positive by construction.

    Parameters:
    -----------
    mean : float
        Desired mean of the distribution
    cv : float
        Desired coefficient of variation (std/mean)
    rng : np.random.RandomState
        Random state for reproducibility

    Returns:
    --------
    sample : float
        A single sample, clamped to [1e-9, inf) to ensure positivity
    """
    if cv <= 0 or mean <= 0:
        return mean
    std = mean * cv
    # Clamp to a small positive floor: rate constants must be > 0
    return max(rng.normal(mean, std), 1e-9)


def _sample_lognormal(mean: float, cv: float, rng: np.random.RandomState) -> float:
    """
    Sample from a log-normal distribution with specified mean and CV.

    The log-normal is parameterized so that:
        E[X] = mean
        CV(X) = std(X) / E[X] = cv

    Parameters:
    -----------
    mean : float
        Desired mean of the distribution
    cv : float
        Desired coefficient of variation (std/mean)
    rng : np.random.RandomState
        Random state for reproducibility

    Returns:
    --------
    sample : float
        A single sample from the log-normal distribution
    """
    if cv <= 0 or mean <= 0:
        return mean
    # Log-normal parameters from desired mean and CV
    sigma2 = np.log(1 + cv ** 2)
    mu = np.log(mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2))


def sample_threshold(CoF_threshold_nominal: float,
                     CV: float = COF_THRESHOLD_CV_DEFAULT,
                     seed: Optional[int] = None) -> float:
    """
    Sample a per-cell condensation threshold from a log-normal distribution.

    Biological rationale: The condensation threshold varies cell-to-cell due
    to differences in nuclear microenvironment (salt concentration, molecular
    crowding, post-translational modifications of coactivator).

    Parameters:
    -----------
    CoF_threshold_nominal : float
        Nominal (population-mean) condensation threshold [CoF*]
    CV : float
        Coefficient of variation for threshold distribution.
        CV=0 returns the nominal value (no heterogeneity).
        Default: 0.25 (25% variability)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    threshold : float
        Sampled threshold for this cell
    """
    if CV <= 0:
        return CoF_threshold_nominal
    rng = np.random.RandomState(seed)
    return _sample_lognormal(CoF_threshold_nominal, CV, rng)


def sample_rate_parameters(k2_base: float, k3_low: float, k3_max: float,
                           kt_base: float, PR_max: float,
                           koff_2: float, koff_3: float,
                           k2_CV: float = K2_BASE_CV_DEFAULT,
                           k3_low_CV: float = K3_LOW_CV_DEFAULT,
                           k3_max_CV: float = K3_MAX_CV_DEFAULT,
                           kt_CV: float = KT_BASE_CV_DEFAULT,
                           PR_CV: float = PR_MAX_CV_DEFAULT,
                           koff_2_CV: float = KOFF_2_CV_DEFAULT,
                           koff_3_CV: float = KOFF_3_CV_DEFAULT,
                           seed: Optional[int] = None) -> Dict[str, float]:
    """
    Sample per-cell rate parameters from log-normal distributions.

    Biological rationale: Even genetically identical cells differ in chromatin
    state, enhancer accessibility, and expression levels of rate-limiting
    factors. On-rates (driven by factor availability) show more variability
    than off-rates (driven by molecular chemistry and structural constraints).

    Parameters:
    -----------
    k2_base : float
        Nominal base rate for 1->2 transition (per unit [TF])
    k3_low : float
        Nominal k3 rate in dilute phase (per unit [CoF])
    k3_max : float
        Nominal maximum k3 in condensed phase
    kt_base : float
        Nominal base transcription rate
    PR_max : float
        Nominal partition ratio in condensed phase
    koff_2 : float
        Nominal off-rate for 2->1 transition
    koff_3 : float
        Nominal off-rate for 3->1 or 3->2 transition
    k2_CV : float
        CV for k2_base sampling (default: 0.3)
    k3_low_CV : float
        CV for k3_low sampling (default: 0.3)
    k3_max_CV : float
        CV for k3_max sampling (default: 0.3)
    kt_CV : float
        CV for kt_base sampling (default: 0.25)
    PR_CV : float
        CV for PR_max sampling (default: 0.2)
    koff_2_CV : float
        CV for koff_2 sampling (default: 0.0, no variability)
    koff_3_CV : float
        CV for koff_3 sampling (default: 0.0, no variability)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    params : dict
        Dictionary with sampled values for each parameter:
        'k2_base', 'k3_low', 'k3_max', 'kt_base', 'PR_max', 'koff_2', 'koff_3'
    """
    rng = np.random.RandomState(seed)

    return {
        'k2_base': _sample_lognormal(k2_base, k2_CV, rng),
        'k3_low': _sample_lognormal(k3_low, k3_low_CV, rng),
        'k3_max': _sample_lognormal(k3_max, k3_max_CV, rng),
        'kt_base': _sample_lognormal(kt_base, kt_CV, rng),
        'PR_max': _sample_lognormal(PR_max, PR_CV, rng),
        'koff_2': _sample_lognormal(koff_2, koff_2_CV, rng),
        'koff_3': _sample_lognormal(koff_3, koff_3_CV, rng),
    }

def sample_normal_parameters(k2_base: float, k3_low: float, k3_max: float,
                           kt_base: float, PR_max: float,
                           koff_2: float, koff_3: float,
                           k2_CV: float = K2_BASE_CV_DEFAULT,
                           k3_low_CV: float = K3_LOW_CV_DEFAULT,
                           k3_max_CV: float = K3_MAX_CV_DEFAULT,
                           kt_CV: float = KT_BASE_CV_DEFAULT,
                           PR_CV: float = PR_MAX_CV_DEFAULT,
                           koff_2_CV: float = KOFF_2_CV_DEFAULT,
                           koff_3_CV: float = KOFF_3_CV_DEFAULT,
                           seed: Optional[int] = None) -> Dict[str, float]:
    """
    Sample per-cell rate parameters from log-normal distributions.

    Biological rationale: Even genetically identical cells differ in chromatin
    state, enhancer accessibility, and expression levels of rate-limiting
    factors. On-rates (driven by factor availability) show more variability
    than off-rates (driven by molecular chemistry and structural constraints).

    Parameters:
    -----------
    k2_base : float
        Nominal base rate for 1->2 transition (per unit [TF])
    k3_low : float
        Nominal k3 rate in dilute phase (per unit [CoF])
    k3_max : float
        Nominal maximum k3 in condensed phase
    kt_base : float
        Nominal base transcription rate
    PR_max : float
        Nominal partition ratio in condensed phase
    koff_2 : float
        Nominal off-rate for 2->1 transition
    koff_3 : float
        Nominal off-rate for 3->1 or 3->2 transition
    k2_CV : float
        CV for k2_base sampling (default: 0.3)
    k3_low_CV : float
        CV for k3_low sampling (default: 0.3)
    k3_max_CV : float
        CV for k3_max sampling (default: 0.3)
    kt_CV : float
        CV for kt_base sampling (default: 0.25)
    PR_CV : float
        CV for PR_max sampling (default: 0.2)
    koff_2_CV : float
        CV for koff_2 sampling (default: 0.0, no variability)
    koff_3_CV : float
        CV for koff_3 sampling (default: 0.0, no variability)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    params : dict
        Dictionary with sampled values for each parameter:
        'k2_base', 'k3_low', 'k3_max', 'kt_base', 'PR_max', 'koff_2', 'koff_3'
    """
    rng = np.random.RandomState(seed)

    return {
        'k2_base': _sample_normal(k2_base, k2_CV, rng),
        'k3_low': _sample_normal(k3_low, k3_low_CV, rng),
        'k3_max': _sample_normal(k3_max, k3_max_CV, rng),
        'kt_base': _sample_normal(kt_base, kt_CV, rng),
        'PR_max': _sample_normal(PR_max, PR_CV, rng),
        'koff_2': _sample_normal(koff_2, koff_2_CV, rng),
        'koff_3': _sample_normal(koff_3, koff_3_CV, rng),
    }


def sample_condensate_localization(P_gene_localization: float,
                                   CoF_conc: float,
                                   CoF_threshold: float,
                                   squelching_factor: Optional[float] = SQUELCHING_FACTOR_DEFAULT,
                                   seed: Optional[int] = None) -> bool:
    """
    Sample whether a condensate forms at the gene locus vs elsewhere.

    Biological rationale: Above the condensation threshold, condensates form
    throughout the nucleus, not exclusively at gene loci. A fraction of cells
    will have condensates away from DNA, leaving the gene in the dilute phase
    even though bulk [CoF] is above threshold.

    Parameters:
    -----------
    P_gene_localization : float
        Base probability that condensate forms at the gene (0 to 1).
        Default: 0.85 (85% chance condensate is at gene)
    CoF_conc : float
        Current coactivator concentration
    CoF_threshold : float
        Condensation threshold for this cell
    squelching_factor : float or None
        If not None, P_gene decreases at high [CoF] to model squelching.
        Not implemented in this version (reserved for future use).
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    at_gene : bool
        True if condensate forms at the gene locus, False if off-gene
    """
    # Below threshold: no condensate, localization is irrelevant
    # (rates will be dilute-phase regardless)
    if CoF_conc < CoF_threshold:
        return True  # No condensate to mislocalize

    rng = np.random.RandomState(seed)
    return rng.uniform() < P_gene_localization


def jitter_concentrations(conc_value: float, n_cells: int,
                          jitter_CV: float = 0.1,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate per-cell jittered concentration values around a nominal value.

    In flow cytometry experiments, cells at the "same" condition actually show
    spread in their measured intensity (x-axis). This function adds log-normal
    jitter to concentration values so scatter plots fill in the gaps between
    discrete grid points, producing a more realistic continuous appearance.

    Parameters:
    -----------
    conc_value : float
        Nominal concentration for this condition
    n_cells : int
        Number of jittered values to generate
    jitter_CV : float
        Coefficient of variation for the jitter (default: 0.1).
        Controls the spread around the nominal value.
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    jittered : np.ndarray
        Array of length n_cells with jittered concentration values
    """
    if jitter_CV <= 0:
        return np.full(n_cells, conc_value)
    rng = np.random.RandomState(seed)
    sigma2 = np.log(1 + jitter_CV ** 2)
    mu = np.log(conc_value) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2), size=n_cells)


def generate_heterogeneous_parameters(
        n_cells: int,
        # Nominal rate parameters
        k2_base: float, k3_low: float, k3_max: float,
        kt_base: float, PR_max: float,
        koff_2: float, koff_3: float,
        CoF_threshold: float,
        # CV parameters
        k2_CV: float = K2_BASE_CV_DEFAULT,
        k3_low_CV: float = K3_LOW_CV_DEFAULT,
        k3_max_CV: float = K3_MAX_CV_DEFAULT,
        kt_CV: float = KT_BASE_CV_DEFAULT,
        PR_CV: float = PR_MAX_CV_DEFAULT,
        koff_2_CV: float = KOFF_2_CV_DEFAULT,
        koff_3_CV: float = KOFF_3_CV_DEFAULT,
        CoF_threshold_CV: float = COF_THRESHOLD_CV_DEFAULT,
        # Localization
        P_gene_localization: float = P_GENE_LOCALIZATION_DEFAULT,
        # Random seed
        base_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate heterogeneous parameters for a population of cells.

    This is a convenience function that calls sample_rate_parameters,
    sample_threshold, and sample_condensate_localization for each cell
    in a population. Useful for pre-generating all per-cell parameters
    before running simulations.

    Parameters:
    -----------
    n_cells : int
        Number of cells (trajectories) to generate parameters for
    k2_base, k3_low, k3_max, kt_base, PR_max : float
        Nominal on-rate parameters
    koff_2, koff_3 : float
        Nominal off-rate parameters
    CoF_threshold : float
        Nominal condensation threshold
    k2_CV, k3_low_CV, k3_max_CV, kt_CV, PR_CV : float
        CVs for on-rate parameters
    koff_2_CV, koff_3_CV : float
        CVs for off-rate parameters (default 0 = no variability)
    CoF_threshold_CV : float
        CV for threshold variability
    P_gene_localization : float
        Base probability condensate forms at gene
    base_seed : int or None
        Base random seed (each cell uses base_seed + cell_index)

    Returns:
    --------
    cell_params : dict
        Dictionary mapping parameter names to arrays of length n_cells:
        'k2_base', 'k3_low', 'k3_max', 'kt_base', 'PR_max',
        'koff_2', 'koff_3', 'CoF_threshold'
    """
    cell_params = {
        'k2_base': np.zeros(n_cells),
        'k3_low': np.zeros(n_cells),
        'k3_max': np.zeros(n_cells),
        'kt_base': np.zeros(n_cells),
        'PR_max': np.zeros(n_cells),
        'koff_2': np.zeros(n_cells),
        'koff_3': np.zeros(n_cells),
        'CoF_threshold': np.zeros(n_cells),
    }

    for i in range(n_cells):
        seed_i = None if base_seed is None else base_seed + i

        # Sample rate parameters
        rates = sample_rate_parameters(
            k2_base, k3_low, k3_max, kt_base, PR_max,
            koff_2, koff_3,
            k2_CV=k2_CV, k3_low_CV=k3_low_CV, k3_max_CV=k3_max_CV,
            kt_CV=kt_CV, PR_CV=PR_CV,
            koff_2_CV=koff_2_CV, koff_3_CV=koff_3_CV,
            seed=seed_i
        )
        for key in rates:
            cell_params[key][i] = rates[key]

        # Sample threshold (use offset seed to avoid correlation with rates)
        threshold_seed = None if base_seed is None else base_seed + n_cells + i
        cell_params['CoF_threshold'][i] = sample_threshold(
            CoF_threshold, CV=CoF_threshold_CV, seed=threshold_seed
        )

    return cell_params
