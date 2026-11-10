"""Extra experiments for the computational-complexity part of the report (independent of main.py)."""
import itertools
import numpy as np
import pandas as pd
import settings as S
from step1_predict import make_features, train_and_validate
from step2_scenarios import sample_scenarios, reduce_kmeans, reduce_random
from step3_optimize import solve, max_cliques, gap
from step4_evaluate import conflicts_per_day

# Steps 1-2 exactly as in main.py
rng = np.random.default_rng(S.SEED)
history, day = pd.read_csv(f"{S.DATA}/history.csv"), pd.read_csv(f"{S.DATA}/test_day.csv")
outcomes = pd.read_csv(f"{S.DATA}/test_outcomes.csv").pivot(index="day", columns="flight", values="arr_delay")
sched_arr, sched_dep = day.sched_arr.to_numpy(), day.sched_dep.to_numpy()
_, residual, model = train_and_validate(history)
raw = sample_scenarios(model.predict(make_features(day)), history, residual, day.weather[0], S.N_RAW_SCENARIOS, rng)
act_arr = sched_arr + outcomes[day.flight].to_numpy()
act_dep = S.departure(act_arr, sched_dep)


def run(delays, probs, formulation="aggregated"):
    """Solve the stochastic model for one scenario set and score the plan on the 300 unseen days."""
    arr_s = sched_arr + delays
    result = solve(sched_arr, sched_dep, arr_s, S.departure(arr_s, sched_dep), probs, formulation)
    if "gate" in result:
        result["conflicts"] = conflicts_per_day(result["gate"], act_arr, act_dep)[0].mean()
    return {k: v for k, v in result.items() if k != "gate"}


# A. Clique constraints versus one constraint per overlapping pair
overlapping = sum(gap(sched_arr, sched_dep, i, j) < 0 for i, j in itertools.combinations(range(len(day)), 2))
cliques = max_cliques(sched_arr, sched_dep)
print(f"A. no-overlap rules per gate: {overlapping} pairwise vs {len(cliques)} clique constraints\n")

# B. How many scenarios, which reduction, which formulation?
rows = []
for K in [5, 10, 20, 50, 100, S.N_RAW_SCENARIOS]:
    if K == S.N_RAW_SCENARIOS:                            # no reduction: all 1000 scenarios
        sets = [("all", 0, raw, np.full(K, 1 / K))]
    else:
        sets = [("k-means", r, *reduce_kmeans(raw, K, r)) for r in range(5)] + \
               [("random", r, *reduce_random(raw, K, np.random.default_rng(r))) for r in range(5)]
    for method, rep, delays, probs in sets:
        rows.append({"K": K, "method": method, "repeat": rep, **run(delays, probs)})
        if rep == 0:                                      # the original scenario-indexed model, once per set type
            rows.append({"K": K, "method": method, "repeat": rep, **run(delays, probs, "scenario")})
study = pd.DataFrame(rows)
study.to_csv(f"{S.RESULTS}/scenario_study.csv", index=False)
print("B. formulation size and time (first repeat)")
print(study[study.repeat == 0].pivot_table(index=["K", "method"], columns="formulation", aggfunc="first",
      values=["variables", "constraints", "seconds", "objective"]).round(3).to_string())
print("\n   out-of-sample conflicts of the aggregated model (mean of 5 repeats)")
print(study[study.formulation == "aggregated"].pivot_table(index="K", columns="method", values="conflicts").round(3), "\n")

# C. Is a bigger fixed buffer in the deterministic model enough?
rows = []
for buffer in [15, 30, 45, 60, 90]:
    plan = solve(sched_arr, sched_dep, sched_arr[None], sched_dep[None], np.ones(1), buffer=buffer)
    rows.append({"buffer": buffer, "conflicts": conflicts_per_day(plan["gate"], act_arr, act_dep)[0].mean(),
                 "apron flights": (plan["gate"] == S.N_GATES).sum()})
buffers = pd.DataFrame(rows)
buffers.to_csv(f"{S.RESULTS}/buffer_study.csv", index=False)
print("C. deterministic plan with a fixed buffer\n", buffers.round(3).to_string(index=False))
