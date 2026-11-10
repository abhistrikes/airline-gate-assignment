"""Step 2 - delay scenarios: residual bootstrap, then k-means reduction (unsupervised learning)."""
import numpy as np
from sklearn.cluster import KMeans


def sample_scenarios(forecast, history, residual, weather, n, rng):
    """n possible days = forecast + a day shock shared by all flights + each flight's own error."""
    h = history.assign(res=residual)
    sd = h.groupby("weather").res.std()                 # thunderstorm days are noisier
    z = h.res / h.weather.map(sd)                       # standardised errors
    shock = z.groupby(h.day).mean()                     # common part of each past day
    own = (z - h.day.map(shock)).to_numpy()             # flight-specific part
    draws = rng.choice(shock.to_numpy(), (n, 1)) + rng.choice(own, (n, len(forecast)))
    return forecast + draws * sd[weather]               # arrival delays, one row per scenario


def reduce_kmeans(scenarios, k, seed):
    """Group similar scenarios; keep the real scenario closest to each centre, weighted by group size."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scenarios)
    kept, probs = [], []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        dist = ((scenarios[members] - km.cluster_centers_[c]) ** 2).sum(axis=1)
        kept.append(scenarios[members[dist.argmin()]])
        probs.append(len(members) / len(scenarios))
    return np.array(kept), np.array(probs)


def reduce_random(scenarios, k, rng):
    """Benchmark: k scenarios picked at random, equal weights."""
    return scenarios[rng.choice(len(scenarios), k, replace=False)], np.full(k, 1 / k)
