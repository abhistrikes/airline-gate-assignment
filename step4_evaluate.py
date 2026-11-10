"""Step 4 - score a gate plan on 300 simulated days that no model has seen."""
import itertools
import numpy as np
import settings as S


def conflicts_per_day(gate, arr, dep):
    """Per day: number of pairs at the same contact gate whose actual ground times overlap."""
    conflicts, overlap_min = np.zeros(len(arr)), np.zeros(len(arr))
    for k in range(S.N_GATES):                            # the apron has room for everyone
        for i, j in itertools.combinations(np.where(gate == k)[0], 2):
            overlap = np.minimum(dep[:, i], dep[:, j]) - np.maximum(arr[:, i], arr[:, j])
            conflicts += overlap > 0
            overlap_min += np.clip(overlap, 0, None)
    return conflicts, overlap_min


def paired_ci(a, b, rng, n_boot=2000):
    """Mean of a - b over the same days, with a 95% bootstrap confidence interval."""
    diff = a - b
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(n_boot)]
    low, high = np.percentile(boot, [2.5, 97.5])
    return diff.mean(), low, high
