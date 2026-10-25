# Predicting Student Academic Performance

A machine-learning group project for **WQD7003 Data Analytics** that explores academic, demographic, and behavioural factors associated with student exam scores. The project is framed around a fictional private tuition-centre stakeholder, *EdTech*, which wants to understand performance drivers and support students who may be at risk of underperforming.

> **Scope of my contribution:** Data modelling — preprocessing pipelines, regression model comparison, cross-validation, hyperparameter tuning, evaluation, and model interpretation.

## Project goals

1. Identify factors associated with student exam performance.
2. Predict `Exam_Score` from available student features.
3. Support data-informed prioritisation of tutoring and academic support.

## Dataset

The supplied notebook expects a CSV named `StudentPerformanceFactors.csv`, described in the group report as a synthetic Kaggle dataset with approximately 6,607 records and 20 features. The CSV was **not included** with the uploaded notebook/report, so it is intentionally not bundled here.

1. Obtain the dataset from the source used by your group and check its licence/usage terms.
2. Place it at `data/raw/StudentPerformanceFactors.csv`.
3. Do not commit sensitive or restricted data to GitHub.

## Repository structure

```text
student-academic-performance-ml/
├── data/
│   └── raw/                 # Place StudentPerformanceFactors.csv here (not tracked)
├── notebooks/
│   └── WQD7003_group_project_original.ipynb
├── reports/
│   └── figures/             # Generated evaluation plots
├── src/
│   ├── train_model.py       # Reproducible modelling pipeline
│   └── predict.py           # Example inference using a saved pipeline
├── .gitignore
├── requirements.txt
└── README.md
```

## Modelling workflow

- Separate predictors from the target (`Exam_Score`).
- Split data into 70% training and 30% test sets with a fixed random seed.
- Impute numeric values with the median and categorical values with the most frequent category **inside the training pipeline**, to avoid fitting preprocessing on the test set.
- Standardise numeric features and one-hot encode categorical features.
- Compare Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting regressors.
- Evaluate with R², MAE, and RMSE; run 10-fold cross-validation for baseline models.
- Tune Ridge, Lasso, ElasticNet, Random Forest, and Gradient Boosting using cross-validated search on the training set.
- Save the selected fitted pipeline and evaluation outputs.
- Estimate feature importance using permutation importance on held-out test data.

The original notebook is preserved as a reference to the group's broader data understanding and EDA work. The script here makes the modelling workflow independently runnable and keeps imputation/encoding/scaling within a scikit-learn pipeline.

## Quick start

Python 3.10+ is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Put the CSV at data/raw/StudentPerformanceFactors.csv

# 4. Train and evaluate
python -m src.train_model
```

Outputs are written to `artifacts/` (created on first run):
- `model_comparison.csv`
- `cross_validation.csv`
- `tuned_model_comparison.csv`
- `permutation_importance.csv`
- `best_model.joblib`
- `metrics.json`

To predict with the saved pipeline, provide a CSV containing the **same predictor columns** as the training data, excluding `Exam_Score`:

```bash
python -m src.predict --input path/to/new_students.csv --output artifacts/predictions.csv
```

## Evaluation metrics

- **R²:** proportion of target variance explained by the model (can be negative on unseen data).
- **MAE:** average absolute prediction error, in exam-score points.
- **RMSE:** square-root of mean squared error; penalises larger errors more than MAE.

For all three metrics, interpret them together rather than relying on a single score. Higher R² is generally better; lower MAE/RMSE are better.

## Interpretation and limitations

The report describes attendance as the leading factor in the Random Forest feature-importance analysis, while Lasso reportedly performed slightly better on overall prediction metrics in the reported comparison. Treat those as **results reported by the submitted group report**, not guaranteed results of this cleaned training script: exact results can vary with preprocessing, split, library versions, and tuning.

Feature importance indicates predictive association, not causation. The dataset is described as synthetic, so findings should not be generalised to real students without external validation. Predictions should support, not replace, professional judgement; avoid using them to stigmatise or penalise students.

## Notes before publishing

- Confirm the dataset's original source and include its citation/licence if required.
- Review the preserved notebook for environment-specific paths and outputs before sharing.
- Add your preferred contact details and any course-required acknowledgements.
- Never commit virtual environments, model binaries, private data, or credentials.
