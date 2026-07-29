"""
Gillespie Algorithm Implementation
==================================

Core simulation engine using the kinetic Monte Carlo (Gillespie) algorithm
for stochastic simulation of chemical reaction systems.

The implementation supports flexible state transition topologies and both
rate-limited and instantaneous transcription modes.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """
    Container for simulation results.
    
    Attributes:
    -----------
    times : np.ndarray
        Array of event times
    states : np.ndarray
        Array of states at each time point
    transcript_counts : np.ndarray
        Cumulative transcript count at each time point
    transcript_times : List[float]
        List of times when transcription events occurred
    """
    times: np.ndarray
    states: np.ndarray
    transcript_counts: np.ndarray
    transcript_times: List[float]


class TwoStateModel:
    """
    Two-state gene expression model: OFF (1) ↔ ON (2)

    State 1: Gene is OFF (unbound or inactive)
    State 2: Gene is ON (actively transcribing)

    Transitions:
    - 1 → 2: rate k_on
    - 2 → 1: rate k_off
    - Transcription in state 2: rate kt (if not instant_transcription)

    Parameters:
    -----------
    k_on : float
        Rate of transition from OFF to ON
    k_off : float
        Rate of transition from ON to OFF
    kt : float
        Transcription rate (transcripts/time) when in state 2
        Ignored if instant_transcription=True
    instant_transcription : bool
        If True, transcription happens instantaneously when entering state 2
        If False, transcription follows Poisson process with rate kt
    """
    
    def __init__(self, k_on: float, k_off: float, kt: float = 1.0, 
                 instant_transcription: bool = False):
        self.k_on = k_on
        self.k_off = k_off
        self.kt = kt
        self.instant_transcription = instant_transcription
        self.n_states = 2
    
    def get_propensities(self, state: int) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate propensities for all possible reactions from current state.
        
        Parameters:
        -----------
        state : int
            Current state (0 or 1)

        Returns:
        --------
        propensities : np.ndarray
            Array of reaction propensities
        reactions : List[str]
            List of reaction identifiers corresponding to propensities
        """
        propensities = []
        reactions = []

        if state == 0:
            # Only state1 → state2 transition possible
            propensities.append(self.k_on)
            reactions.append("0->1")

        elif state == 1:
            # state2 → state1 transition
            propensities.append(self.k_off)
            reactions.append("1->0")
            
            # Transcription (only if not instant mode)
            if not self.instant_transcription:
                propensities.append(self.kt)
                reactions.append("transcribe")
        
        return np.array(propensities), reactions
    
    def execute_reaction(self, state: int, reaction: str) -> Tuple[int, bool]:
        """
        Execute a reaction and return new state and whether transcription occurred.
        
        Parameters:
        -----------
        state : int
            Current state
        reaction : str
            Reaction identifier
            
        Returns:
        --------
        new_state : int
            State after reaction
        transcribed : bool
            Whether a transcript was produced
        """
        if reaction == "0->1":
            # Transition to state 1
            # If instant transcription, produce transcript immediately
            return 1, self.instant_transcription
        elif reaction == "1->0":
            return 0, False
        elif reaction == "transcribe":
            # Transcription event, stay in state 1
            return 1, True
        else:
            raise ValueError(f"Unknown reaction: {reaction}")


class ThreeStateModel:
    """
    Three-state gene expression model with flexible transition topology.

    State 1: OFF - unbound or inactive
    State 2: OFF(2) - TF bound but transcriptionally paused
    State 3: ON - transcriptionally active

    Basic transitions (always present):
    - 1 → 2: rate k2
    - 2 → 1: rate koff_2
    - 2 → 3: rate k3

    Flexible off-transitions from state 3:
    - 3 → 1: rate koff_3_to_1 (if allow_3_to_1=True)
    - 3 → 2: rate koff_3_to_2 (if allow_3_to_2=True)

    Transcription from state 3:
    - If instant_transcription=True: transcribes immediately upon entering state 3
    - If instant_transcription=False: transcribes at rate kt while in state 3

    Parameters:
    -----------
    k2 : float
        Rate of 1 → 2 transition
    koff_2 : float
        Rate of 2 → 1 transition
    k3 : float
        Rate of 2 → 3 transition
    koff_3_to_1 : float
        Rate of 3 → 1 transition (only if allow_3_to_1=True)
    koff_3_to_2 : float
        Rate of 3 → 2 transition (only if allow_3_to_2=True)
    kt : float
        Transcription rate in state 3 (ignored if instant_transcription=True)
    allow_3_to_1 : bool
        Whether 3 → 1 transition is allowed
    allow_3_to_2 : bool
        Whether 3 → 2 transition is allowed
    instant_transcription : bool
        If True, transcription happens instantaneously when entering state 3
    """

    def __init__(self, k2: float, koff_2: float, k3: float,
                 koff_3_to_1: float = 0.0, koff_3_to_2: float = 0.0,
                 kt: float = 1.0,
                 allow_3_to_1: bool = True, allow_3_to_2: bool = False,
                 instant_transcription: bool = False):
        self.k2 = k2
        self.koff_2 = koff_2
        self.k3 = k3
        self.koff_3_to_1 = koff_3_to_1
        self.koff_3_to_2 = koff_3_to_2
        self.kt = kt
        self.allow_3_to_1 = allow_3_to_1
        self.allow_3_to_2 = allow_3_to_2
        self.instant_transcription = instant_transcription
        self.n_states = 3

        # Validate that at least one off-transition from state 3 is allowed
        if not (allow_3_to_1 or allow_3_to_2):
            raise ValueError("At least one off-transition from state 3 must be allowed")
    
    def get_propensities(self, state: int) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate propensities for all possible reactions from current state.
        
        Parameters:
        -----------
        state : int
            Current state (0, 1, or 2)
            
        Returns:
        --------
        propensities : np.ndarray
            Array of reaction propensities
        reactions : List[str]
            List of reaction identifiers
        """
        propensities = []
        reactions = []

        if state == 0:
            # Only state1 → state2 transition
            propensities.append(self.k2)
            reactions.append("0->1")

        elif state == 1:
            # state2 → state1 transition
            propensities.append(self.koff_2)
            reactions.append("1->0")

            # state2 → state3 transition
            propensities.append(self.k3)
            reactions.append("1->2")

        elif state == 2:
            # Off-transitions from state 3
            if self.allow_3_to_1 and self.koff_3_to_1 > 0:
                propensities.append(self.koff_3_to_1)
                reactions.append("2->0")

            if self.allow_3_to_2 and self.koff_3_to_2 > 0:
                propensities.append(self.koff_3_to_2)
                reactions.append("2->1")
            
            # Transcription (only if not instant mode)
            if not self.instant_transcription:
                propensities.append(self.kt)
                reactions.append("transcribe")
        
        return np.array(propensities), reactions
    
    def execute_reaction(self, state: int, reaction: str) -> Tuple[int, bool]:
        """
        Execute a reaction and return new state and whether transcription occurred.
        
        Parameters:
        -----------
        state : int
            Current state
        reaction : str
            Reaction identifier
            
        Returns:
        --------
        new_state : int
            State after reaction
        transcribed : bool
            Whether a transcript was produced
        """
        if reaction == "0->1":
            return 1, False
        elif reaction == "1->0":
            return 0, False
        elif reaction == "1->2":
            # Transition to state 3
            # If instant transcription, produce transcript immediately
            return 2, self.instant_transcription
        elif reaction == "2->0":
            return 0, False
        elif reaction == "2->1":
            return 1, False
        elif reaction == "transcribe":
            # Transcription event, stay in state 3
            return 2, True
        else:
            raise ValueError(f"Unknown reaction: {reaction}")


def gillespie_simulation(model, t_max: float, initial_state: int = 0,
                        seed: Optional[int] = None) -> SimulationResult:
    """
    Run Gillespie simulation for a gene expression model.
    
    This is the core simulation engine that implements the Gillespie algorithm
    (also known as Kinetic Monte Carlo or Stochastic Simulation Algorithm).
    
    Algorithm:
    1. Calculate propensities (rates) for all possible reactions
    2. Sample time to next event from exponential distribution
    3. Choose which reaction occurs based on propensities
    4. Update system state and time
    5. Repeat until t_max is reached
    
    Parameters:
    -----------
    model : TwoStateModel or ThreeStateModel
        Model object containing rate constants and transition topology
    t_max : float
        Maximum simulation time
    initial_state : int
        Starting state (default: 0)
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    result : SimulationResult
        Object containing simulation trajectory and transcript production times
        
    Example:
    --------
    >>> model = ThreeStateModel(k2=0.1, koff_2=0.5, k3=0.2,
    ...                         koff_3_to_1=0.5, kt=2.0)
    >>> result = gillespie_simulation(model, t_max=1000.0, seed=42)
    >>> print(f"Total transcripts: {result.transcript_counts[-1]}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize simulation
    t = 0.0
    state = initial_state
    n_transcripts = 0
    
    # Storage for trajectory
    times = [0.0]
    states = [state]
    transcript_counts = [0]
    transcript_times = []
    
    # Main simulation loop
    while t < t_max:
        # Get propensities for all possible reactions from current state
        propensities, reactions = model.get_propensities(state)
        
        # Calculate total propensity
        total_propensity = np.sum(propensities)
        
        if total_propensity == 0:
            # No more reactions possible (shouldn't happen with well-defined models)
            break
        
        # Sample time to next event from exponential distribution
        # tau ~ Exp(total_propensity)
        tau = np.random.exponential(1.0 / total_propensity)
        t += tau
        
        if t > t_max:
            # Exceeded simulation time
            break
        
        # Choose which reaction occurs
        # Sample uniformly and find which reaction interval it falls into
        r = np.random.uniform(0, total_propensity)
        cumsum = np.cumsum(propensities)
        reaction_idx = np.searchsorted(cumsum, r)
        chosen_reaction = reactions[reaction_idx]
        
        # Execute the chosen reaction
        state, transcribed = model.execute_reaction(state, chosen_reaction)
        
        # Record transcription event if it occurred
        if transcribed:
            n_transcripts += 1
            transcript_times.append(t)
        
        # Record current state
        times.append(t)
        states.append(state)
        transcript_counts.append(n_transcripts)
    
    return SimulationResult(
        times=np.array(times),
        states=np.array(states),
        transcript_counts=np.array(transcript_counts),
        transcript_times=transcript_times
    )


def gillespie_simulation_windowed(
    model_factory: Callable,
    t_max: float,
    window_size: float,
    localization_params: Dict,
    initial_state: int = 0,
    seed: Optional[int] = None
) -> Tuple[SimulationResult, List[bool]]:
    """
    Run Gillespie simulation with periodic re-evaluation of condensate localization.

    This function segments the trajectory into windows and re-samples whether the
    condensate is at the gene or off-gene at each window boundary. Rate constants
    are updated accordingly, but the gene state (OFF/paused/ON) is preserved across
    windows to ensure continuity.

    Parameters:
    -----------
    model_factory : Callable
        Function that takes (at_gene: bool, seed: int) and returns a model object
        with updated rate constants based on the localization state
    t_max : float
        Maximum simulation time
    window_size : float
        Time interval for re-evaluating condensate localization.
        If window_size >= t_max, behaves like standard gillespie_simulation
    localization_params : Dict
        Parameters for localization sampling, must include:
        - 'P_gene_localization': float (probability of being at gene)
        - 'CoF_conc': float (coactivator concentration)
        - 'CoF_threshold': float (condensation threshold)
        - 'localization_sampler': Callable that returns bool (at_gene)
    initial_state : int
        Starting state (default: 0)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    result : SimulationResult
        Combined simulation trajectory across all windows
    localization_states : List[bool]
        Boolean list indicating at_gene status for each window

    Example:
    --------
    >>> def model_factory(at_gene, seed):
    ...     k2, k3, kt = calculate_rates_condensate(..., P_at_gene=at_gene)
    ...     return ThreeStateModel(k2=k2, koff_2=0.1, k3=k3, koff_3_to_2=0.1, kt=kt)
    >>> result, loc_states = gillespie_simulation_windowed(
    ...     model_factory, t_max=1000.0, window_size=25.0,
    ...     localization_params={...}, seed=42
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate number of windows
    n_windows = int(np.ceil(t_max / window_size))

    # Initialize storage
    all_times = []
    all_states = []
    all_transcript_counts = []
    all_transcript_times = []
    localization_states = []

    # Initialize simulation state
    current_state = initial_state
    current_transcript_count = 0
    current_time = 0.0

    # Get the localization sampler function
    localization_sampler = localization_params['localization_sampler']

    # Run simulation for each window
    for window_idx in range(n_windows):
        # Determine window boundaries
        window_start = window_idx * window_size
        window_end = min((window_idx + 1) * window_size, t_max)
        window_duration = window_end - window_start

        # Sample condensate localization for this window
        window_seed = seed + window_idx * 1000 if seed is not None else None
        at_gene = localization_sampler(seed=window_seed)
        localization_states.append(at_gene)

        # Create model with appropriate rates for this localization state
        model = model_factory(at_gene=at_gene, seed=window_seed)

        # Run Gillespie simulation for this window
        result = gillespie_simulation(
            model,
            t_max=window_duration,
            initial_state=current_state,
            seed=window_seed + 100 if window_seed is not None else None
        )

        # Adjust times to account for previous windows
        adjusted_times = result.times + window_start
        adjusted_transcript_times = [t + window_start for t in result.transcript_times]

        # Adjust transcript counts to account for previous windows
        adjusted_counts = result.transcript_counts + current_transcript_count

        # Store results (skip first point if not first window to avoid duplication)
        if window_idx == 0:
            all_times.extend(adjusted_times)
            all_states.extend(result.states)
            all_transcript_counts.extend(adjusted_counts)
        else:
            all_times.extend(adjusted_times[1:])
            all_states.extend(result.states[1:])
            all_transcript_counts.extend(adjusted_counts[1:])

        all_transcript_times.extend(adjusted_transcript_times)

        # Update state for next window
        current_state = result.states[-1]
        current_transcript_count = adjusted_counts[-1]
        current_time = window_end

    # Create combined result
    combined_result = SimulationResult(
        times=np.array(all_times),
        states=np.array(all_states),
        transcript_counts=np.array(all_transcript_counts),
        transcript_times=all_transcript_times
    )

    return combined_result, localization_states


def run_multiple_simulations(model, t_max: float, n_trajectories: int,
                            initial_state: int = 0,
                            base_seed: Optional[int] = None) -> List[SimulationResult]:
    """
    Run multiple independent Gillespie simulations.

    This is useful for computing ensemble averages and statistical properties.

    Parameters:
    -----------
    model : TwoStateModel or ThreeStateModel
        Model object
    t_max : float
        Maximum simulation time for each trajectory
    n_trajectories : int
        Number of independent simulations to run
    initial_state : int
        Starting state for all simulations
    base_seed : int or None
        Base random seed (each trajectory uses base_seed + trajectory_index)

    Returns:
    --------
    results : List[SimulationResult]
        List of simulation results, one per trajectory

    Example:
    --------
    >>> model = TwoStateModel(k_on=0.1, k_off=0.5, kt=2.0)
    >>> results = run_multiple_simulations(model, t_max=1000.0,
    ...                                     n_trajectories=100, base_seed=42)
    >>> mean_transcripts = np.mean([r.transcript_counts[-1] for r in results])
    """
    results = []
    for i in range(n_trajectories):
        seed = None if base_seed is None else base_seed + i
        result = gillespie_simulation(model, t_max, initial_state, seed)
        results.append(result)

    return results
