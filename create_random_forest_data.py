from adaXT.random_forest import RandomForest
from adaXT.criteria import Gini_index, Entropy, Squared_error
from multiprocessing import cpu_count
import numpy as np
import json


def get_regression_data(
    n, m, random_state: np.random.RandomState, lowx=0, highx=100, lowy=0, highy=5
):
    X = random_state.uniform(lowx, highx, (n, m))
    Y = random_state.uniform(lowy, highy, n)
    return (X, Y)


def get_classification_data(
    n, m, random_state: np.random.RandomState, lowx=0, highx=100, lowy=0, highy=5
):
    X = random_state.uniform(lowx, highx, (n, m))
    Y = random_state.randint(lowy, highy, n)
    return (X, Y)


def run_gini_index(X, Y, n_jobs, n_estimators, seed):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Gini_index,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        bootstrap=True,
        max_samples=5,
        random_state=seed,
    )
    forest.fit(X, Y)
    return forest


def run_entropy(X, Y, n_jobs, n_estimators, seed):
    forest = RandomForest(
        forest_type="Classification",
        criteria=Entropy,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        bootstrap=True,
        max_samples=5,
        random_state=seed,
    )
    forest.fit(X, Y)
    return forest


def run_squared_error(X, Y, n_jobs, n_estimators, seed):
    forest = RandomForest(
        forest_type="Regression",
        criteria=Squared_error,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        bootstrap=True,
        max_samples=5,
        random_state=seed,
    )
    forest.fit(X, Y)
    return forest


def create_data():
    random_state = np.random.RandomState(2024)
    seed = 2024
    n = 100
    m = 10
    n_estimators = 100
    X_cla, Y_cla = get_classification_data(n, m, random_state=random_state)
    X_reg, Y_reg = get_regression_data(n, m, random_state=random_state)
    gini_forest = run_gini_index(
        X_cla,
        Y_cla,
        n_jobs=cpu_count(),
        n_estimators=n_estimators,
        seed=seed,
    )
    entropy_forest = run_entropy(
        X_cla,
        Y_cla,
        n_jobs=cpu_count(),
        n_estimators=n_estimators,
        seed=seed,
    )
    squared_forest = run_squared_error(
        X_reg, Y_reg, n_jobs=cpu_count(), n_estimators=n_estimators, seed=seed
    )

    d = dict()
    d["gini_pred"] = gini_forest.predict(X_cla).tolist()
    d["entropy_pred"] = entropy_forest.predict(X_cla).tolist()
    d["squared_pred"] = squared_forest.predict(X_reg).tolist()
    with open("./tests/data/forestData.json", "w") as f:
        json.dump(d, f, indent=3)


if __name__ == "__main__":
    create_data()
