"""Run the whole study: predict -> scenarios -> optimise four plans -> score them on 300 unseen days."""
import os
import numpy as np
import pandas as pd
import settings as S
from step1_predict import make_features, train_and_validate
from step2_scenarios import sample_scenarios, reduce_kmeans
from step3_optimize import solve
from step4_evaluate import conflicts_per_day, paired_ci

os.makedirs(S.RESULTS, exist_ok=True)
rng = np.random.default_rng(S.SEED)
history = pd.read_csv(f"{S.DATA}/history.csv")
day = pd.read_csv(f"{S.DATA}/test_day.csv")
outcomes = pd.read_csv(f"{S.DATA}/test_outcomes.csv").pivot(index="day", columns="flight", values="arr_delay")
sched_arr, sched_dep = day.sched_arr.to_numpy(), day.sched_dep.to_numpy()

# Step 1: learn delays from the past days and forecast the planning day
accuracy, residual, model = train_and_validate(history)
forecast = model.predict(make_features(day))
print("Step 1 - forecast accuracy (minutes, 5-fold CV by day)\n", accuracy.round(2), "\n")

# Step 2: 1000 possible versions of the day, and 20 representative ones chosen by k-means
raw = sample_scenarios(forecast, history, residual, day.weather[0], S.N_RAW_SCENARIOS, rng)
km_delays, km_probs = reduce_kmeans(raw, S.N_SCENARIOS, S.SEED)
all_arr, km_arr, pto_arr = sched_arr + raw, sched_arr + km_delays, sched_arr + forecast
all_dep, km_dep, pto_dep = (S.departure(a, sched_dep) for a in (all_arr, km_arr, pto_arr))

# Step 3: four plans from the same model, each given different information about delays
one, equal = np.ones(1), np.full(S.N_RAW_SCENARIOS, 1 / S.N_RAW_SCENARIOS)
plans = {"Deterministic": solve(sched_arr, sched_dep, sched_arr[None], sched_dep[None], one),   # timetable only
         "Predict-then-optimise": solve(pto_arr, pto_dep, pto_arr[None], pto_dep[None], one),  # point forecast
         "ETO, 20 k-means": solve(sched_arr, sched_dep, km_arr, km_dep, km_probs),            # reduced scenarios
         "ETO, all 1000": solve(sched_arr, sched_dep, all_arr, all_dep, equal)}               # all scenarios

# Step 4: score every plan on the same 300 unseen days
act_arr = sched_arr + outcomes[day.flight].to_numpy()
act_dep = S.departure(act_arr, sched_dep)
daily, rows = pd.DataFrame({"day": outcomes.index}), []
for name, plan in plans.items():
    conflicts, minutes = conflicts_per_day(plan["gate"], act_arr, act_dep)
    daily[name] = conflicts
    expected, _ = conflicts_per_day(plan["gate"], all_arr, all_dep)     # what the 1000 scenarios predict
    rows.append({"plan": name, "mean conflicts": conflicts.mean(), "expected by scenarios": expected.mean(),
                 "days without conflict": (conflicts == 0).mean(),
                 "overlap minutes": minutes.mean(), "apron flights": (plan["gate"] == S.N_GATES).sum(),
                 "variables": plan["variables"], "constraints": plan["constraints"],
                 "solve seconds": plan["seconds"], "status": plan["status"]})
summary = pd.DataFrame(rows).set_index("plan")
for name in list(plans)[1:]:                       # paired difference to the deterministic plan, same days
    diff, low, high = paired_ci(daily[name].to_numpy(), daily["Deterministic"].to_numpy(), rng)
    summary.loc[name, "vs deterministic"], summary.loc[name, "CI low"], summary.loc[name, "CI high"] = diff, low, high
print("Step 4 - out-of-sample results on 300 days\n", summary.round(3).to_string())

# Save everything for the report and the figures
accuracy.to_csv(f"{S.RESULTS}/accuracy.csv")
summary.to_csv(f"{S.RESULTS}/summary.csv")
daily.to_csv(f"{S.RESULTS}/daily_conflicts.csv", index=False)
pd.DataFrame({"flight": day.flight, "sched_arr": sched_arr, "sched_dep": sched_dep, "forecast_delay": forecast,
              **{name: plan["gate"] for name, plan in plans.items()}}).to_csv(f"{S.RESULTS}/plans.csv", index=False)
pd.DataFrame(km_delays, columns=day.flight).assign(probability=km_probs).to_csv(f"{S.RESULTS}/scenarios.csv", index=False)

# Show which flights use each gate in two of the plans (gates 1-7 here; plans.csv numbers them 0-6)
clock = lambda minutes: f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"
for name in ["Deterministic", "ETO, all 1000"]:
    print(f"\nGate plan - {name} (arrival-departure in the timetable)")
    for k in range(S.N_GATES + 1):                 # stand N_GATES is the apron
        at_k = day[plans[name]["gate"] == k].sort_values("sched_arr")
        if len(at_k):
            stand = "Apron " if k == S.N_GATES else f"Gate {k + 1}"
            print(f"  {stand}: " + ", ".join(f"{f.flight} {clock(f.sched_arr)}-{clock(f.sched_dep)}" for f in at_k.itertuples()))
print("\nAll four plans are saved in results/plans.csv.")
