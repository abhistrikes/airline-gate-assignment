"""Step 1 - supervised learning: predict the arrival delay of every flight."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def make_features(df):
    """Information known the day before: time of day, weather forecast, weekday, ground time, airline."""
    X = df[["hour", "weather", "day_of_week"]].copy()
    X["sched_ground"] = df.sched_dep - df.sched_arr
    for airline in ["AA", "BB", "CC"]:
        X[f"airline_{airline}"] = (df.airline == airline).astype(int)
    return X


def train_and_validate(history):
    """Compare three models with 5-fold cross-validation; all flights of one day stay in the same fold."""
    X, y = make_features(history), history.arr_delay
    fold = (history.day % 5).to_numpy()                   # days 5, 10, 15, ... form fold 0, and so on
    models = {"Linear regression": LinearRegression(),
              "Gradient boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                             learning_rate=0.05, random_state=0)}
    pred = {"Weather average": np.zeros(len(y)), **{name: np.zeros(len(y)) for name in models}}
    for f in range(5):
        train, test = fold != f, fold == f                # learn on 4/5 of the days, test on the other days
        average = y[train].groupby(X.weather[train]).mean()          # naive benchmark
        pred["Weather average"][test] = X.weather[test].map(average)
        for name, model in models.items():
            pred[name][test] = model.fit(X[train], y[train]).predict(X[test])
    accuracy = pd.DataFrame({name: {"MAE": mean_absolute_error(y, p),
                                    "RMSE": np.sqrt(np.mean((y - p) ** 2))} for name, p in pred.items()}).T
    residual = (y - pred["Gradient boosting"]).to_numpy()   # errors on days the model did not see
    final_model = models["Gradient boosting"].fit(X, y)     # refit on all past days for the forecast
    return accuracy, residual, final_model
