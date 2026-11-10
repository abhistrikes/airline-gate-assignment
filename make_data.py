"""Create the synthetic data set. The data folder already holds its output; run again only to rebuild it.

Every random part follows a named probability distribution:
  arrivals      Poisson process from 06:00, 2.7 flights per hour (time between arrivals exponential, mean 22.5 min)
  ground time   discrete uniform, 40 to 90 minutes
  airline       categorical: AA 50%, BB 30%, CC 20%
  weather       categorical: fine 50%, rain 30%, thunderstorm 20%
  day shock     normal, mean 0, SD 10 minutes (x1.5 on rain days, x2 on thunderstorm days)
  own delay     gamma (shape 2, scale 6) minus 12: mean 0, with a long tail of very late flights
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N_FLIGHTS, N_HISTORY_DAYS, N_TEST_DAYS = 40, 120, 300
ARRIVALS_PER_HOUR = 40 / 15                           # on average 40 flights between 06:00 and 21:00
AIRLINE_EFFECT = {"AA": 2.0, "BB": 6.0, "CC": -2.0}   # some airlines are later than others

# 1. The daily timetable: the same 40 flights arrive and leave every day
gaps = rng.exponential(60 / ARRIVALS_PER_HOUR, N_FLIGHTS)    # Poisson arrivals = exponential gaps between them
timetable = pd.DataFrame({"flight": [f"F{i + 1:02d}" for i in range(N_FLIGHTS)],
                          "airline": rng.choice(list(AIRLINE_EFFECT), N_FLIGHTS, p=[0.5, 0.3, 0.2]),
                          "sched_arr": (6 * 60 + np.cumsum(gaps)).round()})
timetable["sched_dep"] = timetable.sched_arr + rng.integers(40, 91, N_FLIGHTS)   # ground time incl. padding, uniform
timetable["hour"] = (timetable.sched_arr // 60).astype(int)


def true_delays(weather, day_of_week):
    """Arrival delays (days x flights) of the hidden 'true' process that no model sees."""
    hour = timetable.hour.to_numpy()
    base = timetable.airline.map(AIRLINE_EFFECT).to_numpy() + 0.9 * (hour - 6)    # delays build up during the day
    storm = weather[:, None] * (4 + 18 / (1 + np.exp(-(hour - 15) / 1.2)))     # storms hit hardest after 15:00
    spread = (1 + 0.5 * weather)[:, None]                         # storms make delays more variable
    day_shock = rng.normal(0, 10, (len(weather), 1)) * spread       # shared by every flight of the day
    own_noise = (rng.gamma(2.0, 6.0, (len(weather), N_FLIGHTS)) - 12) * spread   # right-skewed
    return base + storm + 5 * (day_of_week[:, None] == 4) + day_shock + own_noise


def long_table(delays, weather, day_of_week):
    rows = []
    for d in range(len(delays)):
        rows.append(timetable.assign(day=d + 1, day_of_week=day_of_week[d], weather=weather[d],
                                     arr_delay=delays[d].round(1)))
    return pd.concat(rows, ignore_index=True)


# 2. History: 120 past days with mixed weather (0 = fine, 1 = rain, 2 = thunderstorms)
w_hist = rng.choice([0, 1, 2], N_HISTORY_DAYS, p=[0.5, 0.3, 0.2])
dow_hist = np.arange(N_HISTORY_DAYS) % 7
history = long_table(true_delays(w_hist, dow_hist), w_hist, dow_hist)
history.to_csv("data/history.csv", index=False)

# 3. The day to plan: a Friday with a thunderstorm forecast
timetable.assign(day_of_week=4, weather=2).to_csv("data/test_day.csv", index=False)

# 4. 300 possible versions of that day, used only to score the plans
w_test, dow_test = np.full(N_TEST_DAYS, 2), np.full(N_TEST_DAYS, 4)
outcomes = long_table(true_delays(w_test, dow_test), w_test, dow_test)
outcomes[["day", "flight", "arr_delay"]].to_csv("data/test_outcomes.csv", index=False)
print("history:", history.shape, " test day:", len(timetable), " outcomes:", outcomes.shape)
