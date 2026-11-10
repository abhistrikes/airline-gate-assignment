"""All settings of the study in one place."""
import numpy as np

DATA, RESULTS = "data", "results"   # input and output folders
SEED = 2026                         # random seed, so every run gives the same numbers

N_GATES = 7          # contact gates (= peak number of aircraft on the ground); a flight can also be sent to the apron (remote stand)
MIN_TURN = 35        # minimum ground time of an aircraft (minutes)
BUFFER = 15          # idle time we would like between two aircraft at one gate (minutes)
W_CONFLICT = 100     # cost of an expected gate conflict
W_BUFFER = 1         # cost of an expected buffer violation
W_APRON = 60         # cost of using the apron instead of a contact gate

N_RAW_SCENARIOS = 1000   # delay scenarios sampled in step 2
N_SCENARIOS = 20         # scenarios kept after k-means reduction
TIME_LIMIT = 60          # Gurobi time limit per model (seconds)


def departure(arr, sched_dep):
    """An aircraft leaves on time, unless it arrived too late to turn around."""
    return np.maximum(sched_dep, arr + MIN_TURN)
