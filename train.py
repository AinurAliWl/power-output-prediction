"""Compare models, tune on training CV, select on validation, evaluate once on test."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

from data import FEATURES, TARGET, ROOT, SEED, load_data, sha256, split_data

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def candidates():
    def pipe(estimator, polynomial=False):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if polynomial:
            steps += [("polynomial", PolynomialFeatures(degree=2, include_bias=False))]
        if isinstance(estimator, (LinearRegression, Ridge)):
            steps += [("scale", StandardScaler())]
        return Pipeline(steps + [("model", estimator)])
    return {
        "mean_baseline": (pipe(DummyRegressor(strategy="mean")), {}),
        "linear_regression": (pipe(LinearRegression()), {}),
        "polynomial_ridge": (pipe(Ridge(), polynomial=True), {"model__alpha": [0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (pipe(RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=2)), {
            "model__max_depth": [12, None], "model__min_samples_leaf": [1, 3],
        }),
        "hist_gradient_boosting": (pipe(HistGradientBoostingRegressor(
            max_iter=300, early_stopping=False, random_state=SEED)), {
            "model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
            "model__l2_regularization": [0.0, 1.0],
        }),
    }


def metrics(y, prediction):
    return {
        "rmse_mw": float(root_mean_squared_error(y, prediction)),
        "mae_mw": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
    }


def run_experiment() -> dict:
    frame, quality = load_data()
    train, val, test = split_data(frame)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    rows, fitted, trials = [], {}, []
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        for name, (estimator, grid) in candidates().items():
            print(f"Training {name}...", flush=True)
            if grid:
                search = GridSearchCV(estimator, grid, scoring="neg_root_mean_squared_error", cv=cv,
                                      n_jobs=1, refit=True, return_train_score=False)
                search.fit(X_train, y_train)
                model = search.best_estimator_
                cv_rmse = float(-search.best_score_)
                cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
                best_params = search.best_params_
                for index, params in enumerate(search.cv_results_["params"]):
                    trials.append({"model": name, "parameters": params,
                                   "cv_rmse_mw": float(-search.cv_results_["mean_test_score"][index])})
            else:
                scores = -cross_val_score(estimator, X_train, y_train, cv=cv,
                                          scoring="neg_root_mean_squared_error", n_jobs=1)
                model = clone(estimator).fit(X_train, y_train)
                cv_rmse, cv_std, best_params = float(scores.mean()), float(scores.std()), {}
            val_metrics = metrics(y_val, model.predict(X_val))
            row = {"model": name, "cv_rmse_mw": cv_rmse, "cv_std_mw": cv_std,
                   "validation_rmse_mw": val_metrics["rmse_mw"],
                   "validation_mae_mw": val_metrics["mae_mw"],
                   "validation_r2": val_metrics["r2"], "best_parameters": best_params}
            rows.append(row); fitted[name] = model
            print(f"  CV RMSE {cv_rmse:.4f}; validation RMSE {val_metrics['rmse_mw']:.4f}", flush=True)

        # Selection and importance use validation only; test is untouched at this stage.
        leaderboard = sorted(rows, key=lambda item: item["validation_rmse_mw"])
        winner = leaderboard[0]["model"]
        validation_model = fitted[winner]
        importance = permutation_importance(validation_model, X_val, y_val,
            scoring="neg_root_mean_squared_error", n_repeats=10, random_state=SEED, n_jobs=1)
        importances = sorted([
            {"feature": name, "rmse_increase_mw": float(importance.importances_mean[i]),
             "std_mw": float(importance.importances_std[i])}
            for i, name in enumerate(FEATURES)
        ], key=lambda item: item["rmse_increase_mw"], reverse=True)

        full_train = pd.concat([train, val])
        final_model = clone(validation_model).fit(full_train[FEATURES], full_train[TARGET])
        test_prediction = final_model.predict(test[FEATURES])
        test_metrics = metrics(test[TARGET], test_prediction)
        baseline = DummyRegressor(strategy="mean").fit(full_train[FEATURES], full_train[TARGET])
        baseline_metrics = metrics(test[TARGET], baseline.predict(test[FEATURES]))
        # Paired bootstrap describes uncertainty on this held-out sample, not future-plant performance.
        rng = np.random.default_rng(SEED)
        squared_error = (test[TARGET].to_numpy() - test_prediction) ** 2
        rmse_samples = np.sqrt(np.mean(squared_error[rng.integers(0, len(test), (500, len(test)))], axis=1))

    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    prediction_rows = test[FEATURES + [TARGET]].copy()
    prediction_rows["prediction_mw"] = test_prediction
    prediction_rows["residual_mw"] = test[TARGET].to_numpy() - test_prediction
    prediction_rows.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    pd.DataFrame(rows).drop(columns="best_parameters").to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(importances).to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    summary = {
        "quality": quality,
        "split": {"seed": SEED, "train": len(train), "validation": len(val), "test": len(test)},
        "selection_rule": "Lowest validation RMSE after 3-fold tuning on training data only",
        "models": leaderboard, "tuning_trials": trials,
        "selected_model": winner, "permutation_importance_on_validation": importances,
        "test_metrics": test_metrics, "test_mean_baseline": baseline_metrics,
        "test_rmse_bootstrap_95pct": [float(x) for x in np.quantile(rmse_samples, [0.025, 0.975])],
        "limitations": ["One plant at full load", "2006-2011 historical observations",
                        "No timestamps; random split is not a temporal forecast evaluation",
                        "Sensor associations are not causal effects"],
    }
    metadata = {
        "model_name": winner, "features": FEATURES, "target": TARGET, "output_unit": "MW",
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "model_sha256": sha256(model_path), "data_csv_sha256": quality["csv_sha256"],
        "training_rows": len(full_train), "evaluation": test_metrics,
        "training_ranges": {c: {"min": float(full_train[c].min()), "max": float(full_train[c].max())} for c in FEATURES},
        "best_parameters": leaderboard[0]["best_parameters"],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Selected:", winner, "Test:", test_metrics, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_experiment()
