"""Train, compare, tune, and interpret student exam-score regression models."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "StudentPerformanceFactors.csv"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET = "Exam_Score"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Obtain StudentPerformanceFactors.csv "
            "from the source used by your group and place it in data/raw/."
        )
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    if data.empty:
        raise ValueError("No rows with a valid numeric Exam_Score were found.")
    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    return {
        "R2": float(r2_score(y_test, predictions)),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = load_data()
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    preprocessor = build_preprocessor(X_train)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    baseline_rows = []
    fitted_baselines = {}
    print(f"Rows: {len(data):,} | Features: {X.shape[1]} | Target: {TARGET}")
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}\n")

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        baseline_rows.append({"Model": name, **metrics, "Fit_seconds": time.perf_counter() - start})
        fitted_baselines[name] = pipe
        print(f"{name:22s} R²={metrics['R2']:.4f}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}")

    baseline_df = pd.DataFrame(baseline_rows).sort_values("R2", ascending=False)
    baseline_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)

    print("\n10-fold cross-validation (R²) on the full dataset:")
    cv_rows = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        scores = cross_val_score(pipe, X, y, cv=10, scoring="r2", n_jobs=-1)
        cv_rows.append({
            "Model": name, "Mean_R2": float(scores.mean()),
            "Std_R2": float(scores.std()), "Min_R2": float(scores.min()),
            "Max_R2": float(scores.max()),
        })
        print(f"{name:22s} {scores.mean():.4f} ± {scores.std():.4f}")
    pd.DataFrame(cv_rows).sort_values("Mean_R2", ascending=False).to_csv(
        ARTIFACTS / "cross_validation.csv", index=False
    )

    # Tune on the training partition only; the test set remains held out.
    search_spaces = {
        "Ridge (Tuned)": (
            Ridge(),
            {"model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
            "grid", 1,
        ),
        "Lasso (Tuned)": (
            Lasso(max_iter=10000),
            {"model__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
            "grid", 1,
        ),
        "ElasticNet (Tuned)": (
            ElasticNet(max_iter=10000),
            {"model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
             "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "grid", 1,
        ),
        "Random Forest (Tuned)": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [100, 200, 300],
             "model__max_depth": [None, 10, 20, 30],
             "model__min_samples_split": [2, 5, 10],
             "model__min_samples_leaf": [1, 2, 4],
             "model__max_features": [1.0, "sqrt", "log2"]},
            "random", 20,
        ),
        "Gradient Boosting (Tuned)": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200, 300],
             "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
             "model__max_depth": [2, 3, 5],
             "model__min_samples_split": [2, 5, 10],
             "model__min_samples_leaf": [1, 2, 4],
             "model__subsample": [0.7, 0.8, 0.9, 1.0]},
            "random", 20,
        ),
    }

    tuned_rows = []
    tuned_pipelines = {}
    for name, (estimator, params, search_type, n_iter) in search_spaces.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        if search_type == "grid":
            search = GridSearchCV(pipe, params, cv=5, scoring="r2", n_jobs=-1)
        else:
            search = RandomizedSearchCV(
                pipe, params, n_iter=n_iter, cv=5, scoring="r2",
                n_jobs=-1, random_state=RANDOM_STATE
            )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        tuned_pipelines[name] = search.best_estimator_
        tuned_rows.append({
            "Model": name, "Best_CV_R2": float(search.best_score_),
            **metrics, "Fit_seconds": time.perf_counter() - start,
            "Best_params": json.dumps(search.best_params_, default=str),
        })
        print(f"{name:28s} CV R²={search.best_score_:.4f} | test R²={metrics['R2']:.4f}")

    tuned_df = pd.DataFrame(tuned_rows).sort_values("R2", ascending=False)
    tuned_df.to_csv(ARTIFACTS / "tuned_model_comparison.csv", index=False)

    candidates = {name: pipe for name, pipe in fitted_baselines.items()}
    candidates.update(tuned_pipelines)
    candidate_metrics = {name: evaluate(pipe, X_test, y_test) for name, pipe in candidates.items()}
    best_name = max(candidate_metrics, key=lambda name: candidate_metrics[name]["R2"])
    best_pipeline = candidates[best_name]
    best_metrics = candidate_metrics[best_name]
    joblib.dump(best_pipeline, ARTIFACTS / "best_model.joblib")

    # Permutation importance is computed on the original held-out feature columns.
    perm = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10,
        random_state=RANDOM_STATE, scoring="r2", n_jobs=-1
    )
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_mean": perm.importances_mean,
        "Importance_std": perm.importances_std,
    }).sort_values("Importance_mean", ascending=False)
    importance.to_csv(ARTIFACTS / "permutation_importance.csv", index=False)

    top = importance.head(10).sort_values("Importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["Feature"], top["Importance_mean"], xerr=top["Importance_std"])
    plt.xlabel("Decrease in R² after permutation")
    plt.title(f"Top feature permutation importance — {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "permutation_importance.png", dpi=160)
    plt.close()

    summary = {
        "dataset_rows": int(len(data)),
        "feature_count": int(X.shape[1]),
        "target": TARGET,
        "best_model": best_name,
        "test_metrics": best_metrics,
        "random_state": RANDOM_STATE,
        "test_size": 0.30,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBest model by held-out test R²: {best_name}")
    print(json.dumps(best_metrics, indent=2))
    print(f"\nSaved outputs to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
