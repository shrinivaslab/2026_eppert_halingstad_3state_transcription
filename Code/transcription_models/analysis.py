"""
Analysis Utilities
==================

Functions for analyzing simulation results including state occupancies,
burst statistics, and transcript production.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .gillespie import SimulationResult


def calculate_state_occupancies(result: SimulationResult, 
                                n_states: int = 3) -> Dict[int, float]:
    """
    Calculate the fraction of time spent in each state.
    
    Parameters:
    -----------
    result : SimulationResult
        Output from gillespie_simulation
    n_states : int
        Number of states in the model (2 or 3)
        
    Returns:
    --------
    occupancies : Dict[int, float]
        Dictionary mapping state number to fraction of time spent in that state
        Sum of all values equals 1.0
        
    Example:
    --------
    >>> result = gillespie_simulation(model, t_max=1000.0)
    >>> occ = calculate_state_occupancies(result, n_states=3)
    >>> print(f"State 1: {occ[0]:.3f}, State 2: {occ[1]:.3f}, State 3: {occ[2]:.3f}")
    """
    times = result.times
    states = result.states
    
    # Calculate duration in each state
    state_durations = {i: 0.0 for i in range(n_states)}
    
    for i in range(len(times) - 1):
        dt = times[i+1] - times[i]
        state_durations[states[i]] += dt
    
    # Convert to fractions
    total_time = sum(state_durations.values())
    occupancies = {s: state_durations[s] / total_time for s in state_durations}
    
    return occupancies


# def analyze_bursts(result: SimulationResult, active_state: int = 2,
#                   min_burst_transcripts: int = 1) -> Dict[str, float]:
#     """
#     Analyze transcriptional bursts (periods in the active state).
    
#     A burst is defined as a continuous period spent in the active state
#     (typically state 3 for three-state models, state 2 for two-state models).
    
#     Parameters:
#     -----------
#     result : SimulationResult
#         Output from gillespie_simulation
#     active_state : int
#         Which state is considered "active" for transcription (default: 2)
#     min_burst_transcripts : int
#         Minimum number of transcripts to count as a burst (default: 1)
        
#     Returns:
#     --------
#     burst_stats : Dict[str, float]
#         Dictionary containing:
#         - 'n_bursts': number of bursts
#         - 'mean_burst_duration': average duration of bursts (time units)
#         - 'mean_burst_size': average number of transcripts per burst
#         - 'burst_frequency': bursts per unit time
#         - 'fraction_time_active': fraction of time in bursting state
        
#     Example:
#     --------
#     >>> result = gillespie_simulation(model, t_max=1000.0)
#     >>> stats = analyze_bursts(result, active_state=2)
#     >>> print(f"Burst frequency: {stats['burst_frequency']:.4f} bursts/time")
#     >>> print(f"Mean burst size: {stats['mean_burst_size']:.2f} transcripts")
#     """
#     times = result.times
#     states = result.states
#     transcript_times = result.transcript_times
    
#     # Find all bursts (continuous periods in active state)
#     bursts = []  # List of (start_time, end_time) tuples
#     in_burst = False
#     burst_start = None
    
#     for i in range(len(times)):
#         if states[i] == active_state and not in_burst:
#             # Start of a burst
#             in_burst = True
#             burst_start = times[i]
#         elif states[i] != active_state and in_burst:
#             # End of a burst
#             in_burst = False
#             bursts.append((burst_start, times[i]))
    
#     # If simulation ended during a burst, close it
#     if in_burst:
#         bursts.append((burst_start, times[-1]))
    
#     # Calculate burst durations
#     burst_durations = [end - start for start, end in bursts]
    
#     # Calculate number of transcripts per burst
#     burst_sizes = []
#     for start, end in bursts:
#         n_transcripts = sum(1 for t in transcript_times if start <= t < end)
#         burst_sizes.append(n_transcripts)
    
#     # Filter bursts by minimum size
#     filtered_bursts = [(dur, size) for dur, size in zip(burst_durations, burst_sizes)
#                       if size >= min_burst_transcripts]
    
#     if len(filtered_bursts) == 0:
#         return {
#             'n_bursts': 0,
#             'mean_burst_duration': 0.0,
#             'mean_burst_size': 0.0,
#             'burst_frequency': 0.0,
#             'fraction_time_active': 0.0
#         }
    
#     filtered_durations, filtered_sizes = zip(*filtered_bursts)
    
#     total_time = times[-1] - times[0]
#     total_burst_time = sum(filtered_durations)
    
#     return {
#         'n_bursts': len(filtered_bursts),
#         'mean_burst_duration': np.mean(filtered_durations),
#         'mean_burst_size': np.mean(filtered_sizes),
#         'burst_frequency': len(filtered_bursts) / total_time,
#         'fraction_time_active': total_burst_time / total_time
#     }


"""
Fixed version of analyze_bursts that excludes truncated bursts.

This corrects the systematic bias where high kt conditions showed 
artificially low burst durations due to including incomplete bursts
that were still ongoing when simulation ended.
"""

import numpy as np
from typing import Dict


def analyze_bursts(result, active_state: int = 2,
                         min_burst_transcripts: int = 1,
                         exclude_truncated: bool = True) -> Dict[str, float]:
    """
    Analyze transcriptional bursts (periods in the active state).
    
    A burst is defined as a continuous period spent in the active state.
    
    CORRECTED: Now excludes truncated bursts by default to avoid 
    systematic bias in burst duration measurements.
    
    Parameters:
    -----------
    result : SimulationResult
        Output from gillespie_simulation
    active_state : int
        Which state is considered "active" for transcription (default: 2, i.e.
        paper state 3)
    min_burst_transcripts : int
        Minimum number of transcripts to count as a burst (default: 1)
    exclude_truncated : bool
        If True (default), exclude bursts that were still ongoing when 
        simulation ended. This prevents systematic bias in duration measurements.
        
    Returns:
    --------
    burst_stats : Dict[str, float]
        Dictionary containing:
        - 'n_bursts': number of COMPLETE bursts
        - 'mean_burst_duration': average duration of COMPLETE bursts (time units)
        - 'mean_burst_size': average number of transcripts per burst
        - 'burst_frequency': bursts per unit time
        - 'fraction_time_active': fraction of time in bursting state
    """
    times = result.times
    states = result.states
    transcript_times = result.transcript_times
    
    # Find all bursts (continuous periods in active state)
    bursts = []  # List of (start_time, end_time, is_complete) tuples
    in_burst = False
    burst_start = None
    
    for i in range(len(times)):
        if states[i] == active_state and not in_burst:
            # Start of a burst
            in_burst = True
            burst_start = times[i]
        elif states[i] != active_state and in_burst:
            # End of a burst (COMPLETE - it exited state 3)
            in_burst = False
            bursts.append((burst_start, times[i], True))  # True = complete
    
    # If simulation ended during a burst, it's INCOMPLETE
    if in_burst:
        if not exclude_truncated:
            bursts.append((burst_start, times[-1], False))  # False = incomplete
        # If exclude_truncated=True, we simply don't add it
    
    # Calculate burst durations and sizes
    burst_durations = []
    burst_sizes = []
    complete_bursts = []
    
    for start, end, is_complete in bursts:
        # Only include complete bursts (or all if exclude_truncated=False)
        if is_complete or not exclude_truncated:
            duration = end - start
            n_transcripts = sum(1 for t in transcript_times if start <= t < end)
            
            burst_durations.append(duration)
            burst_sizes.append(n_transcripts)
            complete_bursts.append((duration, n_transcripts))
    
    # Filter bursts by minimum size
    filtered_bursts = [(dur, size) for dur, size in complete_bursts
                      if size >= min_burst_transcripts]
    
    if len(filtered_bursts) == 0:
        return {
            'n_bursts': 0,
            'mean_burst_duration': 0.0,
            'mean_burst_size': 0.0,
            'burst_frequency': 0.0,
            'fraction_time_active': 0.0
        }
    
    filtered_durations, filtered_sizes = zip(*filtered_bursts)
    
    total_time = times[-1] - times[0]
    total_burst_time = sum(filtered_durations)
    
    return {
        'n_bursts': len(filtered_bursts),
        'mean_burst_duration': np.mean(filtered_durations),
        'mean_burst_size': np.mean(filtered_sizes),
        'burst_frequency': len(filtered_bursts) / total_time,
        'fraction_time_active': total_burst_time / total_time
    }



def calculate_transcript_statistics(results: List[SimulationResult],
                                    t_max: float) -> Dict[str, float]:
    """
    Calculate statistics across multiple simulation trajectories.
    
    Parameters:
    -----------
    results : List[SimulationResult]
        List of simulation results from multiple runs
    t_max : float
        Simulation time for each trajectory
        
    Returns:
    --------
    stats : Dict[str, float]
        Dictionary containing:
        - 'mean_transcripts': mean total transcripts across trajectories
        - 'std_transcripts': standard deviation of transcripts
        - 'mean_rate': mean transcription rate (transcripts/time)
        - 'std_rate': standard deviation of transcription rate
        - 'cv_transcripts': coefficient of variation of transcript counts
        
    Example:
    --------
    >>> results = run_multiple_simulations(model, t_max=1000.0, n_trajectories=100)
    >>> stats = calculate_transcript_statistics(results, t_max=1000.0)
    >>> print(f"Mean transcripts: {stats['mean_transcripts']:.1f} ± {stats['std_transcripts']:.1f}")
    >>> print(f"CV: {stats['cv_transcripts']:.3f}")
    """
    transcript_counts = [r.transcript_counts[-1] for r in results]
    rates = [count / t_max for count in transcript_counts]
    
    mean_transcripts = np.mean(transcript_counts)
    std_transcripts = np.std(transcript_counts)
    cv = std_transcripts / mean_transcripts if mean_transcripts > 0 else 0.0
    
    return {
        'mean_transcripts': mean_transcripts,
        'std_transcripts': std_transcripts,
        'mean_rate': np.mean(rates),
        'std_rate': np.std(rates),
        'cv_transcripts': cv
    }


def calculate_state_transition_matrix(result: SimulationResult,
                                     n_states: int = 3) -> np.ndarray:
    """
    Calculate the empirical state transition count matrix.
    
    This counts how many transitions occurred between each pair of states.
    
    Parameters:
    -----------
    result : SimulationResult
        Output from gillespie_simulation
    n_states : int
        Number of states in the model
        
    Returns:
    --------
    transition_matrix : np.ndarray
        Matrix where element [i,j] is the number of transitions from state i to state j
        
    Example:
    --------
    >>> result = gillespie_simulation(model, t_max=1000.0)
    >>> M = calculate_state_transition_matrix(result, n_states=3)
    >>> print(f"Transitions 1→2: {M[0,1]}")
    >>> print(f"Transitions 3→1: {M[2,0]}")
    """
    states = result.states
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i+1]
        if from_state != to_state:  # Only count actual transitions
            transition_matrix[from_state, to_state] += 1
    
    return transition_matrix


def calculate_mean_first_passage_time(results: List[SimulationResult],
                                     from_state: int, to_state: int) -> float:
    """
    Calculate mean first passage time from one state to another.
    
    This measures how long it takes on average to reach to_state starting
    from from_state.
    
    Parameters:
    -----------
    results : List[SimulationResult]
        Multiple simulation trajectories
    from_state : int
        Starting state
    to_state : int
        Target state
        
    Returns:
    --------
    mfpt : float
        Mean first passage time (average over all trajectories)
        
    Example:
    --------
    >>> results = run_multiple_simulations(model, t_max=1000.0, n_trajectories=100)
    >>> mfpt = calculate_mean_first_passage_time(results, from_state=0, to_state=2)
    >>> print(f"Time to reach state 3 from state 1: {mfpt:.2f} time units")
    """
    passage_times = []
    
    for result in results:
        times = result.times
        states = result.states
        
        # Find first time we're in from_state
        in_from_state = False
        start_time = None
        
        for i, (t, s) in enumerate(zip(times, states)):
            if s == from_state and not in_from_state:
                in_from_state = True
                start_time = t
            elif s == to_state and in_from_state:
                # Reached target state
                passage_times.append(t - start_time)
                break
    
    if len(passage_times) == 0:
        return np.nan
    
    return np.mean(passage_times)


def compare_models(results_dict: Dict[str, List[SimulationResult]],
                  t_max: float) -> Dict[str, Dict[str, float]]:
    """
    Compare transcript statistics across different models.
    
    Parameters:
    -----------
    results_dict : Dict[str, List[SimulationResult]]
        Dictionary mapping model names to lists of simulation results
        Example: {'binding': [result1, result2, ...], 'condensate': [...]}
    t_max : float
        Simulation time
        
    Returns:
    --------
    comparison : Dict[str, Dict[str, float]]
        Nested dictionary with statistics for each model
        
    Example:
    --------
    >>> results_binding = run_multiple_simulations(model_binding, t_max=1000, n_trajectories=100)
    >>> results_condensate = run_multiple_simulations(model_condensate, t_max=1000, n_trajectories=100)
    >>> comparison = compare_models(
    ...     {'binding': results_binding, 'condensate': results_condensate},
    ...     t_max=1000.0
    ... )
    >>> print(f"Binding: {comparison['binding']['mean_transcripts']:.1f}")
    >>> print(f"Condensate: {comparison['condensate']['mean_transcripts']:.1f}")
    """
    comparison = {}
    
    for model_name, results in results_dict.items():
        comparison[model_name] = calculate_transcript_statistics(results, t_max)
    
    return comparison
