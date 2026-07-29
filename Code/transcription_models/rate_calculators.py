"""
Rate Calculators
================

Functions for calculating concentration-dependent rate constants.

Two mechanisms are implemented:
1. Simple binding: rates scale linearly with concentration
2. Condensation: sharp threshold behavior with phase transition
"""

import numpy as np
from typing import Tuple


def calculate_rates_binding(TF_conc: float, CoF_conc: float,
                           k2_base: float, k3_base: float,
                           koff_2: float, koff_3: float,
                           kt_base: float, kt_CoF_factor: float = 0.0) -> Tuple[float, float, float]:
    """
    Calculate rates for simple binding model (no condensation).

    In this model, all rates scale linearly with concentrations following
    simple mass-action kinetics. There is no cooperativity or phase transition.

    Rate equations:
    ---------------
    k2 = k2_base * [TF]                    # TF binding drives 1→2 transition
    k3 = k3_base * [CoF]                   # CoF binding drives 2→3 transition
    kt = kt_base + kt_CoF_factor * [CoF]   # Transcription rate (optionally CoF-dependent)

    This represents a scenario where:
    - Transcription factor (TF) recruitment is proportional to TF concentration
    - Coactivator (CoF) recruitment is proportional to CoF concentration
    - No cooperative effects or threshold behavior

    Parameters:
    -----------
    TF_conc : float
        Transcription factor concentration (arbitrary units)
    CoF_conc : float
        Coactivator concentration (arbitrary units)
    k2_base : float
        Base rate constant for 1→2 transition (per unit [TF])
    k3_base : float
        Base rate constant for 2→3 transition (per unit [CoF])
    koff_2 : float
        Off-rate for 2→1 transition (not concentration-dependent)
    koff_3 : float
        Off-rate for 3→1 or 3→2 transition (not concentration-dependent)
    kt_base : float
        Base transcription rate (concentration-independent component)
    kt_CoF_factor : float, optional
        Factor for coactivator-dependent transcription enhancement
        (default: 0.0, meaning kt is concentration-independent)

    Returns:
    --------
    k2 : float
        Effective rate for 1→2 transition
    k3 : float
        Effective rate for 2→3 transition
    kt : float
        Effective transcription rate

    Example:
    --------
    >>> k2, k3, kt = calculate_rates_binding(
    ...     TF_conc=1.0, CoF_conc=2.0,
    ...     k2_base=0.1, k3_base=0.2,
    ...     koff_2=0.5, koff_3=0.5,
    ...     kt_base=2.0, kt_CoF_factor=0.0
    ... )
    >>> print(f"k2={k2:.2f}, k3={k3:.2f}, kt={kt:.2f}")
    k2=0.10, k3=0.40, kt=2.00
    """
    # Linear scaling with concentration (mass-action kinetics)
    k2 = k2_base * TF_conc
    k3 = k3_base * CoF_conc
    kt = kt_base + kt_CoF_factor * CoF_conc

    return k2, k3, kt


def calculate_rates_condensate(TF_conc: float, CoF_conc: float,
                               k2_base: float, k3_low: float, k3_max: float,
                               CoF_threshold: float, PR_max: float,
                               koff_2: float, koff_3: float,
                               kt_base: float, kt_CoF_factor: float = 0.0,
                               dense_phase_CoF: float = 15.0,
                               P_at_gene: bool = True,
                               off_gene_dilute_conc: float = None) -> Tuple[float, float, float]:
    """
    Calculate rates for condensation model with phase transition.

    This model implements a sharp threshold behavior representing biomolecular
    condensate formation. Above a critical coactivator concentration, the system
    undergoes a phase transition where:
    - Local TF concentration is amplified by partitioning into condensate
    - CoF-dependent transition rate saturates at maximum value
    - Transcription rate can be enhanced

    The model captures the behavior shown in the diagram where:
    - Below threshold: dilute phase, linear scaling
    - Above threshold: condensed phase, saturation and amplification

    Rate equations:
    ---------------
    If [CoF] < [CoF*] (dilute phase):
        PR = 1.0                              # No partitioning
        k3 = k3_low * [CoF]                   # Linear scaling
        kt = kt_base + kt_CoF_factor * [CoF]  # Linear scaling

    If [CoF] >= [CoF*] (condensed phase) AND P_at_gene=True:
        PR = PR_max                           # TF partitions into condensate
        k3 = k3_max                           # Saturated (threshold reached)
        kt = kt_base + kt_CoF_factor * (effective [CoF] in condensate)

    If [CoF] >= [CoF*] but P_at_gene=False (condensate not at gene):
        Uses dilute phase rates (as if at threshold concentration)

    k2 = k2_base * [TF] * PR                  # Amplified by partition ratio

    Parameters:
    -----------
    TF_conc : float
        Global transcription factor concentration
    CoF_conc : float
        Global coactivator concentration
    k2_base : float
        Base rate constant for 1→2 transition (per unit [TF])
    k3_low : float
        k3 rate in dilute phase (per unit [CoF])
    k3_max : float
        Maximum k3 in condensed phase (saturation value)
    CoF_threshold : float
        Critical coactivator concentration for phase transition [CoF*]
    PR_max : float
        Partition ratio in condensed phase (>1)
        Represents [TF]_local / [TF]_global in condensate
    koff_2 : float
        Off-rate for 2→1 transition
    koff_3 : float
        Off-rate for 3→1 or 3→2 transition
    kt_base : float
        Base transcription rate
    kt_CoF_factor : float, optional
        Factor for coactivator-dependent transcription enhancement
    dense_phase_CoF : float, optional
        Fixed CoF concentration in dense phase (default: 15.0)
    P_at_gene : bool, optional
        Whether the condensate is localized at the gene locus (default: True).
        If False and [CoF] >= threshold, the gene remains in the dilute phase
        and uses dilute-phase rates at the threshold concentration. This models
        the scenario where condensates form elsewhere in the nucleus.
    off_gene_dilute_conc : float or None, optional
        The CoF concentration to use for dilute-phase rate calculations when
        the condensate is NOT at the gene (P_at_gene=False).
        If None (default), uses CoF_threshold (the condition's own threshold).
        Set this to a fixed reference value (e.g. 1.0 = WT threshold) so that
        all threshold conditions produce the same off-gene background rates and
        therefore converge to the same plateau TC above threshold.

    Returns:
    --------
    k2 : float
        Effective rate for 1→2 transition (amplified by condensate)
    k3 : float
        Effective rate for 2→3 transition (saturates above threshold)
    kt : float
        Effective transcription rate

    Notes:
    ------
    The partition ratio (PR) represents the fold-enrichment of TF in the
    condensed phase compared to the dilute phase. For example, PR=2.0 means
    the local TF concentration is 2× the global concentration.

    The kt calculation in condensed phase accounts for the effective CoF
    concentration that would give the same k3_max in the dilute phase.

    Example:
    --------
    >>> # Below threshold
    >>> k2, k3, kt = calculate_rates_condensate(
    ...     TF_conc=1.0, CoF_conc=0.5,
    ...     k2_base=0.1, k3_low=0.1, k3_max=0.5,
    ...     CoF_threshold=1.0, PR_max=2.0,
    ...     koff_2=0.5, koff_3=0.5,
    ...     kt_base=2.0, kt_CoF_factor=0.0
    ... )
    >>> print(f"Below threshold: k2={k2:.2f}, k3={k3:.2f}")
    Below threshold: k2=0.10, k3=0.05

    >>> # Above threshold
    >>> k2, k3, kt = calculate_rates_condensate(
    ...     TF_conc=1.0, CoF_conc=2.0,
    ...     k2_base=0.1, k3_low=0.1, k3_max=0.5,
    ...     CoF_threshold=1.0, PR_max=2.0,
    ...     koff_2=0.5, koff_3=0.5,
    ...     kt_base=2.0, kt_CoF_factor=0.0
    ... )
    >>> print(f"Above threshold: k2={k2:.2f}, k3={k3:.2f}")
    Above threshold: k2=0.20, k3=0.50
    """
    if CoF_conc < CoF_threshold:
        # Below threshold - no condensate
        PR = 1.0
        k3 = k3_low * CoF_conc
        effective_CoF = CoF_conc
    elif not P_at_gene:
        # Above threshold but condensate is NOT at this gene.
        # Gene stays in dilute phase. The effective local CoF concentration
        # is the reference dilute-phase value — either the caller-specified
        # off_gene_dilute_conc or, if not set, this condition's own threshold.
        # Using a fixed off_gene_dilute_conc (e.g. WT threshold = 1.0) across
        # all threshold conditions ensures the off-gene background is identical
        # regardless of which threshold is being swept, so plateau TCs converge.
        dilute_conc = off_gene_dilute_conc if off_gene_dilute_conc is not None \
                      else CoF_threshold
        PR = 1.0
        k3 = k3_low * dilute_conc
        effective_CoF = dilute_conc
    else:
        # Above threshold - condensate forms at gene
        # Dense phase properties are FIXED
        PR = PR_max
        k3 = k3_max
        effective_CoF = dense_phase_CoF  # FIXED VALUE

    k2 = k2_base * TF_conc * PR
    kt = kt_base + kt_CoF_factor * effective_CoF

    return k2, k3, kt

    # # Check if we're above or below the condensation threshold
    # if CoF_conc < CoF_threshold:
    #     # DILUTE PHASE: below threshold
    #     # No partitioning, linear scaling with concentration
    #     PR = 1.0
    #     k3 = k3_low * CoF_conc
    #     kt = kt_base + kt_CoF_factor * CoF_conc
    # else:
    #     # CONDENSED PHASE: above threshold
    #     # TF partitions into condensate, k3 saturates
    #     PR = PR_max
    #     k3 = k3_max

    #     # Calculate effective CoF concentration in condensate
    #     # This is the [CoF] that would give k3_max in dilute phase
    #     effective_CoF = k3_max / (k3_low * CoF_threshold) if (k3_low * CoF_threshold) > 0 else 0
    #     kt = kt_base + kt_CoF_factor * effective_CoF

    # # k2 is amplified by the partition ratio
    # # In condensed phase, local [TF] is PR times higher than global [TF]
    # k2 = k2_base * TF_conc * PR

    # return k2, k3, kt


def sweep_concentrations(TF_range: np.ndarray, CoF_range: np.ndarray,
                        rate_calculator: callable,
                        **rate_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate rates across a range of TF and CoF concentrations.
    
    This is useful for generating phase diagrams and understanding how
    rates vary across concentration space.
    
    Parameters:
    -----------
    TF_range : np.ndarray
        Array of TF concentrations to evaluate
    CoF_range : np.ndarray
        Array of CoF concentrations to evaluate
    rate_calculator : callable
        Either calculate_rates_binding or calculate_rates_condensate
    **rate_params : dict
        Parameters to pass to rate_calculator (k2_base, k3_base, etc.)

    Returns:
    --------
    k2_grid : np.ndarray
        2D array of k2 values, shape (len(TF_range), len(CoF_range))
    k3_grid : np.ndarray
        2D array of k3 values
    kt_grid : np.ndarray
        2D array of kt values

    Example:
    --------
    >>> TF_range = np.logspace(-1, 1, 20)
    >>> CoF_range = np.logspace(-1, 1, 20)
    >>> k2_grid, k3_grid, kt_grid = sweep_concentrations(
    ...     TF_range, CoF_range,
    ...     calculate_rates_condensate,
    ...     k2_base=0.1, k3_low=0.1, k3_max=0.5,
    ...     CoF_threshold=1.0, PR_max=2.0,
    ...     koff_2=0.5, koff_3=0.5,
    ...     kt_base=2.0
    ... )
    """
    n_TF = len(TF_range)
    n_CoF = len(CoF_range)

    k2_grid = np.zeros((n_TF, n_CoF))
    k3_grid = np.zeros((n_TF, n_CoF))
    kt_grid = np.zeros((n_TF, n_CoF))

    for i, TF in enumerate(TF_range):
        for j, CoF in enumerate(CoF_range):
            k2, k3, kt = rate_calculator(TF, CoF, **rate_params)
            k2_grid[i, j] = k2
            k3_grid[i, j] = k3
            kt_grid[i, j] = kt

    return k2_grid, k3_grid, kt_grid
