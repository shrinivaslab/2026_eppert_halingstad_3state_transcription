"""
Transcription Models Package
============================

A flexible package for simulating stochastic gene expression models using
the Gillespie algorithm (kinetic Monte Carlo).

State Indexing Convention:
--------------------------
States are labeled 1, 2, 3 to match the manuscript:
- State 1: OFF (unbound or inactive)
- State 2: OFF (TF bound but transcriptionally paused)
- State 3: ON (transcriptionally active)

Rate constants and model parameters are named after these labels (e.g. `k2` is the
rate into state 2, `koff_3_to_1` is the rate from state 3 to state 1). Internally,
Python arrays and dict keys still use 0-based positions to hold per-state
data — a plain implementation detail that only matters if you're reading the
simulation engine's source directly.

Models Available:
----------------
- Two-state model: OFF (1) ↔ ON (2)
- Three-state models with various transition topologies:
  - Standard: 1 ↔ 2 ↔ 3, with 3 → 1
  - State 3 → State 2 only
  - State 3 → State 1 only
  - State 3 → both State 2 and State 1

Mechanisms:
-----------
- Simple binding: rates scale linearly with concentration
- Condensation: sharp threshold behavior above critical concentration
"""

from .gillespie import (
    gillespie_simulation,
    gillespie_simulation_windowed,
    run_multiple_simulations,
    TwoStateModel,
    ThreeStateModel
)
from .rate_calculators import (
    calculate_rates_binding,
    calculate_rates_condensate
)
from .analysis import (
    calculate_state_occupancies,
    analyze_bursts,
    calculate_transcript_statistics
)
from .heterogeneity import (
    sample_threshold,
    sample_rate_parameters,
    sample_condensate_localization,
    generate_heterogeneous_parameters,
    jitter_concentrations,
)
from .sweep import run_sweep, run_full_sweep

__version__ = "0.1.0"
__all__ = [
    'gillespie_simulation',
    'gillespie_simulation_windowed',
    'run_multiple_simulations',
    'TwoStateModel',
    'ThreeStateModel',
    'calculate_rates_binding',
    'calculate_rates_condensate',
    'calculate_state_occupancies',
    'analyze_bursts',
    'calculate_transcript_statistics',
    'sample_threshold',
    'sample_rate_parameters',
    'sample_condensate_localization',
    'generate_heterogeneous_parameters',
    'jitter_concentrations',
    'run_sweep',
    'run_full_sweep',
]
