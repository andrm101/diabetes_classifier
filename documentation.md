## 0. Data Engineering
Already Done

### 0.1 Data loading & integrity
- `X_train.csv` (53,019 × 21 features), `y_train.csv`, `X_test.csv` (17,673 rows), joined on `ID` (one-to-one merge, `ID` used as index).
- No missing values, no duplicate IDs in train or test.
- Target is balanced: Yes = 26,510, No = 26,509.
- Only non-numeric column: `Gender` (`Male`/`Female`), encoded with a fixed map (Female = 0, Male = 1).

### 0.2 Outlier analysis (Tukey fences, 1.5 × IQR)
- Applied to the 7 non-binary features only. For binary 0/1 features the IQR is either 0 (every minority value gets flagged) or 1 (nothing can be flagged), so the check is meaningless there.

| Feature | Fences | Outliers | % rows |
|---|---|---|---|
| PhysHlth | [-7.5, 12.5] | 9,923 | 18.7 |
| MentHlth | [-3, 5] | 8,750 | 16.5 |
| BMI | [13, 45] | 1,625 | 3.1 |
| GenHlth, Age, Education, Income | — | 0 | 0 |

- 15,342 unique IDs (28.9 %) flagged in at least one column. IDs per feature are stored in `outlier_ids`, boxplots in `outputs/figures/boxplots_non_binary.png`.
- **Decision: no rows removed.** The flagged values are valid survey answers (0–30 days, plausible BMI) and they carry predictive signal: % diabetes is 78.3 for BMI > 45, 70.6 for PhysHlth > 12.5 and 59.9 for MentHlth > 5, against 50.0 overall. Dropping them would bias the sample toward healthy respondents and break the class balance.

### 0.3 Transformations
- `log1p` on BMI, MentHlth, PhysHlth (fixed transform, no leakage):

| Feature | Skew raw → log | Excess kurtosis raw → log | % IQR outliers raw → log |
|---|---|---|---|
| BMI | 1.72 → 0.57 | 7.02 → 1.08 | 3.1 → 1.8 |
| PhysHlth | 1.67 → 0.89 | 1.25 → −0.79 | 18.7 → 0.0 |
| MentHlth | 2.43 → 1.40 | 4.70 → 0.48 | 16.5 → 11.1 |

- The skew left in MentHlth/PhysHlth comes from the zeros (68.1 % and 56.5 % of rows report 0 days), not from the tails. Among rows with at least one bad day, the log values are symmetric (skew 0.07 and −0.11), so no monotone transform can remove it.

### 0.4 Feature engineering: two-part (hurdle) encoding
- `engineer_features()` produces `X_fe` (23 columns), fixed rules only, so it can be reapplied to `X_test` unchanged:
  - `MentHlth_any`, `PhysHlth_any`: 1 if at least one bad day was reported, else 0
  - `BMI_log`, `MentHlth_log`, `PhysHlth_log`: log1p of the raw value (the raw columns are dropped)
- After encoding, the amount part has skew 0.07 / −0.11 and 0 % outliers.
- The indicators carry signal: % diabetes is 60.5 vs 41.9 for PhysHlth_any = 1 vs 0, and 53.0 vs 48.6 for MentHlth_any.

### 0.5 Scaler comparison (log1p features + scaler on the remaining columns, 5-fold stratified CV)


## 1. Feature Selection 
EDA -> Feature Selection
KBEST, Recursive Feature Elimination, PCA -> Feature Selection

### 1.1 EDA after feature engineering
- Figures: `dist_non_binary_fe.png` (per-class histograms), `prevalence_binary_fe.png` (per-class prevalence of binary features).
- Clear class separation: GenHlth, Age (shifted up for diabetes), BMI_log (shifted right), HighBP (75 % vs 37 %), HighChol, DiffWalk, and a spike at 30 days in PhysHlth_log.
- Lower in the diabetes class: Income, Education, PhysActivity, Veggies, Fruits, HvyAlcoholConsump.
- Little separation: AnyHealthcare and CholCheck (95–99 % in both classes), Gender, MentHlth_any.

### 1.2 Correlation (Spearman)
- Only 3 pairs with |ρ| > 0.5: MentHlth_log–MentHlth_any (0.98) and PhysHlth_log–PhysHlth_any (0.95), both by construction, plus PhysHlth_log–GenHlth (0.52).
- VIF: only the hurdle `_log` columns slightly exceed 5 (5.28 / 5.13). All original features are < 2, so there is no multicollinearity problem.

### 1.3 Filter scores (SelectKBest: mutual information, χ²)
- All methods agree on the strong features: GenHlth, HighBP, Age, HighChol, DiffWalk, PhysHlth_log, Income, HeartDiseaseorAttack.
- Weakest: AnyHealthcare (the only non-significant one, χ² p = 0.32), MentHlth_any, Gender, NoDocbcCost, Fruits.
- BMI_log ranks 3rd by MI but 19th by χ². χ² assumes count/categorical data and is distorted by the log scale, so MI is used for continuous features.

### 1.4 Method choice
- **PCA rejected:** it is feature extraction, not selection, so interpretability is lost. It ignores the target, suits mostly binary/ordinal data poorly, and is not needed at 23 features / 53k rows (~2,300 rows per feature). It would also re-merge the `_any`/`_log` pairs.
- **SelectKBest:** univariate only (ignores redundancy), so it is used as an exploratory screen.
- **RFE(CV):** multivariate, but the selection is tied to the eliminating model, so it is run per model (section 2.1).

## 2. Choose Models
LOGREG
DT
RF
SVM
AdaBoost
GradBoost
XGBoost
Neural Networks 

Parameter Tuning -> GridSearch

Differentiated, Overfitted Model OR Loose Underfitted Catch-em All

### 2.1 Baselines: RFE-selected vs. all 23 features (default hyperparameters, 5-fold stratified CV)
- RFECV per model with the model itself as the eliminator. The RFE variant is re-validated with RFE inside the pipeline, so elimination is redone in every fold.
- Preprocessing: StandardScaler for LogReg and SVM, raw features for tree models. SVM = `LinearSVC`, because an RBF kernel has no importances for RFE and is too slow at 53k rows.

| Model | All 23: Acc / AUC / F1 | RFE (k): Acc / AUC / F1 | k (1-SE rule) |
|---|---|---|---|
| LogReg | 0.753 / 0.829 / 0.758 | (20) 0.753 / 0.829 / 0.758 | 8 |
| DecisionTree | 0.658 / 0.658 / 0.656 | (3) 0.724 / 0.795 / 0.732 | 3 |
| RandomForest | 0.742 / 0.815 / 0.750 | optimum = all 23 | 20 |
| SVM (linear) | 0.753 / 0.829 / 0.759 | (18) 0.753 / 0.829 / 0.759 | 9 |
| AdaBoost | 0.749 / 0.826 / 0.755 | (10) 0.749 / 0.826 / 0.755 | 7 |
| GradBoost | 0.756 / 0.835 / 0.765 | (12) 0.756 / 0.834 / 0.765 | 10 |
| XGBoost | 0.750 / 0.826 / 0.759 | (12) 0.754 / 0.829 / 0.763 | 10 |

- For 5 of 7 models, RFE ≈ all features (Δ ≤ 0.0003, fold-std ≈ 0.003–0.004).
- The exceptions are over-flexible defaults. The DecisionTree has no depth limit and overfits, so cutting it to 3 features (GenHlth, Age, BMI_log) acts as crude regularisation. XGBoost (depth 6, learning rate 0.3) gains +0.004, about 1 std. Both are expected to close under tuning.
- The tree ensembles consistently drop Smoker, Stroke, PhysActivity, Fruits, Veggies, AnyHealthcare, NoDocbcCost.
- **Decision:** tune all models on all 23 features. RFE adds 3–10× fit time for no reliable gain.
- Curves: `rfecv_curves_all_models.png`. Results: `outputs/cv_results.csv` (experiment = `baseline_rfe_vs_all`).

### 2.2 Hyperparameter tuning (all 23 features, 5-fold stratified CV, refit on accuracy)
- GridSearchCV: LogReg, SVM (linear), DecisionTree, AdaBoost. RandomizedSearchCV: RandomForest (30 candidates), GradBoost (30), XGBoost (40).

| Model | Baseline acc | Tuned acc (± std) | Tuned AUC | Tuned F1 | Δ acc | Best parameters |
|---|---|---|---|---|---|---|
| XGBoost | 0.750 | **0.758 ± 0.003** | **0.835** | **0.768** | +0.008 | lr 0.019, 759 trees, depth 6, min_child_weight 10, subsample 0.73, colsample 0.66 |
| GradBoost | 0.756 | **0.758 ± 0.002** | **0.835** | **0.768** | +0.002 | lr 0.029, 198 trees, depth 5, subsample 0.85 |
| RandomForest | 0.742 | 0.755 ± 0.003 | 0.832 | 0.765 | +0.013 | 539 trees, depth 12, min_samples_leaf 5, max_features sqrt |
| LogReg | 0.753 | 0.753 ± 0.004 | 0.829 | 0.758 | 0.000 | C 0.1, L2 |
| AdaBoost | 0.749 | 0.753 ± 0.003 | 0.831 | 0.758 | +0.003 | lr 1.0, 400 estimators (grid edge) |
| SVM (linear) | 0.753 | 0.753 ± 0.003 | 0.829 | 0.759 | 0.000 | C 0.01 |
| DecisionTree | 0.658 | 0.746 ± 0.003 | 0.821 | 0.753 | +0.087 | depth 9, min_samples_leaf 100, gini |

- **Best:** XGBoost and GradBoost are tied (0.758, Δ 0.0002 ≪ std). Tuning closed the gaps flagged in 2.1: the DecisionTree went from 0.658 to 0.746 once its depth was limited, and XGBoost gained more than RFE gave it.
- **Linear models are already at their ceiling with defaults** (Δ = 0): the problem has a mostly linear ~0.753 floor, and the trees add about +0.005 through interactions/non-linearity.
- **Caveat:** the best CV score is slightly optimistic, because parameters were chosen on the same folds. AdaBoost's best values sit at the grid edge (largest lr and n_estimators), so it may gain a little from a wider grid.
- Files: `outputs/best_params.json`, `outputs/cv_results.csv` (experiment = `tuned`), `tuning_baseline_vs_tuned.png`, `feature_importance_XGBoost_tuned.png`.
