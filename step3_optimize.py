"""Step 3 - stochastic gate assignment model (after Seker & Noyan, 2012), solved with Gurobi."""
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import settings as S


def max_cliques(arr, dep):
    """Groups of flights that are all on the ground at one moment (maximal cliques of the interval graph)."""
    cliques = []
    for t in sorted(arr):                                 # a new group can only start when a flight arrives
        on_ground = frozenset(np.where((arr <= t) & (dep > t))[0])
        if not any(on_ground <= c for c in cliques):
            cliques = [c for c in cliques if not c <= on_ground] + [on_ground]
    return [c for c in cliques if len(c) > 1]


def gap(arr, dep, i, j):
    """Idle minutes between flights i and j at one gate (negative = their ground times overlap)."""
    return np.maximum(arr[..., j] - dep[..., i], arr[..., i] - dep[..., j])


def pair_cost(g, buffer):
    """Cost if two flights with idle time g share a gate: a conflict (g < 0) or a buffer violation."""
    return S.W_CONFLICT * (g < 0) + S.W_BUFFER * ((g >= 0) & (g < buffer))


def solve(nom_arr, nom_dep, arr_s, dep_s, probs, formulation="aggregated", buffer=S.BUFFER):
    """Plan on nominal times (no overlap allowed) and minimise the expected cost over the scenarios."""
    n, G = len(nom_arr), S.N_GATES                        # stand index G is the apron
    m = gp.Model()
    m.Params.OutputFlag, m.Params.TimeLimit = 0, S.TIME_LIMIT
    x = m.addVars(n, G + 1, vtype=GRB.BINARY)             # x[i,k] = 1 if flight i uses stand k
    m.addConstrs(x.sum(i, "*") == 1 for i in range(n))    # every flight gets exactly one stand
    for Q in max_cliques(nom_arr, nom_dep):               # the nominal plan must be conflict-free
        m.addConstrs(gp.quicksum(x[i, k] for i in Q) <= 1 for k in range(G))
    # only pairs that do not overlap nominally but come too close in at least one scenario matter
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if gap(nom_arr, nom_dep, i, j) >= 0 and (gap(arr_s, dep_s, i, j) < buffer).any()]
    terms = []
    if formulation == "aggregated":                       # this study: one variable per pair
        y = m.addVars(pairs, ub=1)                        # y[i,j] = 1 if i and j share a contact gate
        m.addConstrs(y[i, j] >= x[i, k] + x[j, k] - 1 for i, j in pairs for k in range(G))
        for i, j in pairs:                                # expected cost of the pair, summed over scenarios
            terms.append(probs @ pair_cost(gap(arr_s, dep_s, i, j), buffer) * y[i, j])
    else:                                                 # original form: one variable per pair AND scenario
        for s, (i, j) in itertools.product(range(len(probs)), pairs):
            cost_s = pair_cost(gap(arr_s[s], dep_s[s], i, j), buffer)
            if cost_s > 0:
                c = m.addVar(ub=1)
                m.addConstrs(c >= x[i, k] + x[j, k] - 1 for k in range(G))
                terms.append(probs[s] * cost_s * c)
    m.setObjective(gp.quicksum(terms) + S.W_APRON * x.sum("*", G), GRB.MINIMIZE)
    m.update()
    result = {"formulation": formulation, "variables": m.NumVars, "constraints": m.NumConstrs}
    try:
        m.optimize()
    except gp.GurobiError as err:                         # e.g. too big for the free pip licence
        return {**result, "status": "too big for licence" if "size-limited" in str(err) else str(err)}
    if m.SolCount == 0:
        return {**result, "status": "no solution in time"}
    gate = np.array([max(range(G + 1), key=lambda k: x[i, k].X) for i in range(n)])
    return {**result, "status": "optimal" if m.Status == GRB.OPTIMAL else "time limit", "gate": gate,
            "objective": m.ObjVal, "bound": m.ObjBound, "seconds": m.Runtime}
