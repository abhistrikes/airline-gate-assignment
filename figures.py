"""Draw the figures of the report from the CSV files in data/ and results/ (run main.py and experiments.py first)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import settings as S

R, OUT = S.RESULTS, f"{S.RESULTS}/figures"
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLAN_COLOR = {"Deterministic": "#2a78d6", "Predict-then-optimise": "#eb6834",
              "ETO, 20 k-means": "#1baf7a", "ETO, all 1000": "#eda100"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": "#8a8984", "legend.frameon": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "savefig.dpi": 200})


K_TICKS = [5, 10, 20, 50, 100, 1000]
COMMAS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)


# 0. The generated data: Poisson arrivals and delays by weather
day, history = pd.read_csv(f"{S.DATA}/test_day.csv"), pd.read_csv(f"{S.DATA}/history.csv")
rate = len(day) / 15                                       # flights per hour used in make_data.py
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
hours = np.arange(6, 21)
axes[0].bar(hours, [(day.hour == h).sum() for h in hours], color="#8a8984", width=0.7)
axes[0].axhline(rate, color=INK, ls="--", lw=1.2, label=f"Poisson mean {rate:.1f}")
axes[0].set_xticks(hours[::2], [f"{h:02d}" for h in hours[::2]]), axes[0].set_xlabel("hour of the day")
axes[0].set_title("Arrivals per hour"), axes[0].grid(axis="x", visible=False), axes[0].set_ylim(0, 6.8)
axes[0].legend(loc="upper left")
gaps, edges = np.diff(day.sched_arr), np.arange(0, 91, 10)
axes[1].hist(gaps, bins=edges, color="#8a8984", edgecolor="white", linewidth=1.5)
t = np.linspace(0, 90, 200)
axes[1].plot(t, len(gaps) * 10 * rate / 60 * np.exp(-t * rate / 60), color=INK, ls="--", lw=1.2,
             label=f"exponential, mean {60 / rate:.1f}")
axes[1].set_xlabel("minutes"), axes[1].set_title("Time between arrivals"), axes[1].legend()
for w, (name, color) in enumerate([("fine", "#1baf7a"), ("rain", "#2a78d6"), ("thunderstorm", "#eb6834")]):
    axes[2].hist(history.arr_delay[history.weather == w], bins=np.arange(-40, 141, 10), density=True,
                 histtype="step", lw=2, color=color, label=name)
axes[2].set_xlabel("minutes"), axes[2].set_yticks([]), axes[2].legend()
axes[2].set_title("Arrival delay in the history")
save(fig, "fig7_data")

# 1. Forecast accuracy of the three models
acc = pd.read_csv(f"{R}/accuracy.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
for ax, metric in zip(axes, ["MAE", "RMSE"]):
    ax.barh(acc.index, acc[metric], color=["#b5b4ae", "#8a8984", INK2], height=0.6)
    for y, v in enumerate(acc[metric]):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", color=INK)
    ax.set_title(f"{metric} (minutes)"), ax.grid(axis="y", visible=False), ax.invert_yaxis()
save(fig, "fig1_accuracy")

# 2. Gate charts: the deterministic plan and the plan that uses all scenarios
plans = pd.read_csv(f"{R}/plans.csv")
fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
for ax, name in zip(axes, ["Deterministic", "ETO, all 1000"]):
    for _, f in plans.iterrows():
        gate = f[name] + 1
        ax.barh(gate, f.sched_dep - f.sched_arr, left=f.sched_arr, height=0.62, color=PLAN_COLOR[name],
                edgecolor="white", linewidth=1.5)
        ax.text((f.sched_arr + f.sched_dep) / 2, gate, f.flight[1:], ha="center", va="center", fontsize=7, color="white")
    tight = sum(((g.sched_arr.shift(-1) - g.sched_dep) < 30).sum()
                for _, g in plans.sort_values("sched_arr").groupby(name))
    ax.set_title(f"{name}: {tight} gap{'' if tight == 1 else 's'} under 30 minutes between flights at one gate")
    ax.set_yticks(range(1, S.N_GATES + 1)), ax.set_ylabel("Gate"), ax.grid(axis="y", visible=False)
axes[1].set_xticks(range(360, 1321, 120), [f"{h:02d}:00" for h in range(6, 23, 2)])
axes[1].set_xlim(350, 1340)
save(fig, "fig2_gate_plans")

# 3. Out-of-sample conflicts of the four plans
daily = pd.read_csv(f"{R}/daily_conflicts.csv").drop(columns="day")
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for ax, values, title in [(axes[0], daily.mean(), "Mean conflicts per day (95% CI)"),
                          (axes[1], (daily == 0).mean() * 100, "Days without any conflict (%)")]:
    err = 1.96 * daily.std() / np.sqrt(len(daily)) if ax is axes[0] else None
    ax.bar(range(4), values, yerr=err, color=[PLAN_COLOR[p] for p in daily], width=0.65, capsize=3,
           error_kw={"ecolor": INK2, "elinewidth": 1})
    for x, v in enumerate(values):
        ax.text(x, v + (0 if err is None else err.iloc[x]) + values.max() * 0.02, f"{v:.2f}" if err is not None else f"{v:.0f}",
                ha="center", va="bottom", color=INK)
    ax.set_xticks(range(4), [p.replace(", ", ",\n").replace("Predict-then-optimise", "Predict-then-\noptimise") for p in daily],
                  fontsize=8.5)
    ax.set_title(title), ax.grid(axis="x", visible=False), ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
save(fig, "fig3_conflicts")

# 4. Model size against the number of scenarios
study = pd.read_csv(f"{R}/scenario_study.csv")
size = study[study.repeat == 0].drop_duplicates(["K", "formulation"])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, col in zip(axes, ["variables", "constraints"]):
    for form, color, label in [("aggregated", "#2a78d6", "Aggregated (this study)"),
                               ("scenario", "#eb6834", "Scenario-indexed (Seker & Noyan)")]:
        s = size[size.formulation == form]
        ax.plot(s.K, s[col], "-o", color=color, lw=2, ms=6, label=label)
        ax.annotate(f"{s[col].iloc[-1]:,}", (s.K.iloc[-1], s[col].iloc[-1]), xytext=(-6, 6),
                    textcoords="offset points", ha="right", color=INK, fontsize=8.5)
    ax.axhline(2000, color=INK2, ls="--", lw=1)
    left = ax is axes[0]
    ax.text(5 if left else 1000, 2250, "free licence limit (2,000)", color=INK2, fontsize=8, ha="left" if left else "right")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xticks(K_TICKS), ax.xaxis.set_major_formatter(COMMAS)
    ax.yaxis.set_major_formatter(COMMAS), ax.set_xlabel("Number of scenarios K"), ax.set_title(f"Number of {col}")
axes[0].legend(loc="upper left")
save(fig, "fig4_model_size")

# 5. Plan quality against the number of scenarios: k-means against random choice
agg = study[study.formulation == "aggregated"]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for method, color in [("k-means", "#1baf7a"), ("random", "#e87ba4")]:
    g = agg[agg.method == method].groupby("K").conflicts
    ax.fill_between(g.mean().index, g.min(), g.max(), color=color, alpha=0.18, lw=0)
    ax.plot(g.mean().index, g.mean(), "-o", color=color, lw=2, ms=6, label=f"{method} (mean and range of 5 runs)")
full = agg[agg.method == "all"].conflicts.iloc[0]
ax.plot(S.N_RAW_SCENARIOS, full, "D", color="#eda100", ms=8, label="all 1000 scenarios")
ax.set_xscale("log"), ax.set_xticks(K_TICKS), ax.xaxis.set_major_formatter(COMMAS)
ax.set_xlabel("Number of scenarios K"), ax.legend(), ax.set_ylim(bottom=0)
ax.set_title("Out-of-sample conflicts per day")
save(fig, "fig5_scenario_count")

# 6. A fixed buffer in the deterministic model, against the stochastic plan
buf = pd.read_csv(f"{R}/buffer_study.csv")
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(buf.buffer, buf.conflicts, "-o", color="#2a78d6", lw=2, ms=6, label="Deterministic plan with fixed buffer")
ax.axhline(full, color="#eda100", lw=2, label="ETO, all 1000 scenarios")
ax.set_xticks(buf.buffer), ax.set_xlabel("Buffer required between flights (minutes)"), ax.legend(), ax.set_ylim(bottom=0)
ax.set_title("Out-of-sample conflicts per day")
save(fig, "fig6_buffer")
print("figures saved in", OUT)
