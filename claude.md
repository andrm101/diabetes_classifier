# CLAUDE.md — Kaggle Diabetes ML

Project-level rules. These override the Sandbox-root CLAUDE.md wherever they conflict.

## Role in this project
- Claude writes and runs code here, about 90% of the time. Theory is the user's job.
- Do not explain methods, assumptions, or literature unless asked. No essays, no "why logistic regression works".
- Default output: code change, then 1–3 lines saying what changed and what the result was (score, shape, error).
- Do not get ahead of the user. Do the step that was asked for, then stop. Don't build the next stage of `documentation.md` on your own.
- If a request is ambiguous in a way that changes the code, ask one short question. Otherwise pick the conventional default and say which one you picked.

## Task facts (from overview.md)
- Binary classification: `Diabetes` ∈ {`Yes`, `No`}. The training set is a **balanced** subset of the CDC Diabetes Health Indicators dataset.
- Files (immutable, never edit): `X_train.csv` (~53k rows), `y_train.csv`, `X_test.csv` (~17.7k rows), `sample_submission.csv`.
- Key column: `ID`. Join `X_train`↔`y_train` on `ID`. Never align them by row order.
- Features: 21. All are numeric floats except `Gender` (`Male`/`Female` string).
  - Binary 0/1: HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk
  - Ordinal: GenHlth (1–5), Age (bucketed), Education, Income
  - Count/continuous: BMI, MentHlth (0–30), PhysHlth (0–30)
- The overview mentions "35 features" in one place. The data has 21. Trust the data.
- The evaluation metric is not stated. Assume **accuracy**, since the classes are balanced. Also report ROC-AUC and F1, and switch the primary metric if the user names another.
- Submission: `ID,Diabetes` with string labels `Yes`/`No`. Same IDs and same order as `X_test.csv`.

## Planned pipeline (from documentation.md)
0. Data prep: encode Gender, scale (Standard/MinMax/Robust)
1. EDA → feature selection: SelectKBest, RFE, PCA
2. Models: LogReg, DT, RF, SVM, AdaBoost, GradBoost, XGBoost, NN (MLP)
3. Tuning: GridSearchCV (RandomizedSearchCV for big grids)

## Non-negotiable ML rules
- **No leakage.** Scaling, encoding, feature selection, and PCA go inside an `sklearn.pipeline.Pipeline`/`ColumnTransformer` that is fit within CV folds. Never call `fit_transform` on the full `X_train` before CV.
- Never fit anything on `X_test`. It is touched only by `predict`.
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Use the same splitter for every model so scores are comparable.
- Seeds: `RANDOM_STATE = 42` as a module-level constant, passed to every estimator and splitter that accepts it.
- Report CV results as mean ± std across folds, never as a single split.
- Tree models don't need scaling. Don't add a scaler to their pipelines unless the user asks.
- SVM on 53k rows is slow (roughly O(n²) or worse). Tune on a stratified subsample or use `LinearSVC`/`SGDClassifier`, and say so.
- Use `n_jobs=-1` for CV and grid search.
- Before the first model run, check for NaNs, duplicate IDs, the label distribution, and dtypes. Report these in one line each.

## Code structure
Keep it flat and small. This is a coursework Kaggle, not a production system.

```
Kaggle_Diabetes_ML/
├── main.py            # entry point: load → CV/compare → fit best → write submission
├── src/
│   ├── data.py        # load_data(), encode, train/test frames keyed on ID
│   ├── features.py    # preprocessing ColumnTransformer + selectors
│   ├── models.py      # model registry + param grids
│   └── evaluate.py    # CV runner, results table, submission writer/validator
├── notebooks/         # EDA only — never the source of truth for the pipeline
├── outputs/
│   ├── cv_results.csv # one row per (model, params): mean/std of each metric
│   ├── figures/
│   └── submissions/   # submission_<model>_<YYYYMMDD-HHMM>.csv
└── (raw CSVs stay at the project root, untouched)
```
- Only split `main.py` into `src/` once it grows past ~200 lines or the user asks. Until then a single well-sectioned `main.py` is fine.
- Paths: `ROOT = Path(__file__).resolve().parent` (or the parent of `src/`). No absolute paths.
- Every function signature has type hints. Public functions get a one-line docstring, no more.
- Don't use `inplace=True` in pandas. Always reassign.
- No magic numbers. Thresholds, k, and grids are named constants or live in `models.py`.
- Figures: matplotlib/seaborn, 300 DPI, axis labels, a title, and a source caption ("Data: CDC Diabetes Health Indicators (Kaggle subset)"). Save to `outputs/figures/`.

## Submission checklist (enforce in code, not by eye)
`write_submission()` must assert:
- the columns are exactly `["ID", "Diabetes"]`
- `len == len(X_test)` and the IDs equal `X_test["ID"]` in order
- the values are a subset of `{"Yes", "No"}`, with no 0/1 and no booleans
- there are no NaNs

It never overwrites an earlier submission. Every file gets a timestamped name.

## Workflow (RTS doctrine, scaled down for a small repo)
- **Scout** inline with Read/Grep. The repo is tiny, so don't spawn subagents to explore it.
- **Build** in the main session for normal tasks. Use parallel subagents only when the user asks, or for genuinely independent heavy jobs (e.g. tuning 3+ model families at once).
- **Verify** before saying something is done: run the script top-to-bottom and show the actual CV numbers or error output. "It should work" doesn't count.
- Keep diffs narrow. Don't refactor code the user didn't ask about.
- Long-running jobs (grid search, SVM, NN) go in the background. Tell the user the expected runtime before launching anything likely to take more than ~2 min.
- Track model comparison in `outputs/cv_results.csv` (append), not in chat.

## Environment
- Python with pandas, numpy, scikit-learn, matplotlib, seaborn. Optional: xgboost.
- Before using xgboost, check that it imports. If it's missing, say so and use `HistGradientBoostingClassifier` as a stand-in. Don't pip install anything without asking.
- The data is ~70k rows total, so pandas is fine. No Spark.

## Don'ts
- Don't modify the raw CSVs.
- Don't tune on the test set or use test predictions to pick models.
- Don't print whole dataframes or long logs into chat. Use summaries only.
- Don't write theory, report prose, or R Markdown unless asked. The Sandbox-root academic-report rules (Rmd, bib, Hausman, etc.) don't apply here.
- Don't make causal claims from feature importances. "Predictive of", never "causes".
