# Exploratory data analysis

Generated directly from `data\raw\processed.cleveland.data`, SHA-256 `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`.
Variant: UCI original processed Cleveland (303 records verified). Source: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/45/heart+disease).

## Overview and quality

Raw shape: 303 × 14. Normalized shape: 303 × 14 (13 predictors and one binary target).
Feature names: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal. Numeric storage is float64 to permit missing values; target is int64. Eight coded features are treated categorically in modeling.
Missing cells by feature: ca: 4, thal: 2. Missing targets are rejected, never imputed.
Exact duplicates removed: 0. No values were silently corrected. 
Original target transformation: 0 → 0; 1,2,3,4 → 1.
Class 0: 164; class 1: 139. Positive fraction: 45.87%; majority/minority ratio: 1.180.

## Numerical observations

| Feature | Minimum | Median | Maximum |
|---|---:|---:|---:|
| age | 29.0 | 56.0 | 77.0 |
| trestbps | 94.0 | 130.0 | 200.0 |
| chol | 126.0 | 241.0 | 564.0 |
| thalach | 71.0 | 153.0 | 202.0 |
| oldpeak | 0.0 | 0.8 | 6.2 |

Pearson correlations with target: oldpeak = 0.425, thalach = -0.417, age = 0.223, trestbps = 0.151, chol = 0.085.
These describe this sample and do not establish causality. Outliers are retained when they satisfy the documented entry schema; imputation is deferred to training folds.

## Figures and detailed inspection

- figures/target_distribution.png: class balance.
- figures/missing_values.png: feature missingness.
- figures/numeric_distributions.png: continuous distributions.
- figures/numeric_by_target.png: class-conditioned measurements.
- figures/correlations.png: continuous-feature relationships (coded categories excluded).
- figures/categorical_distributions.png: all eight categorical features by class.
- descriptive_statistics.csv: counts, mean, SD and quartiles; coded-variable means have limited meaning.
- eda_summary.json: computed types, class counts, correlations, mappings and provenance.

The notebook displays sample records locally. Full raw or processed records are intentionally not copied into reports intended for Git.

## Limitations

This historical hospital cohort is small and selected; missing values and site differences can affect models. Limited sex encoding cannot represent all people. Age and sex composition may create demographic bias; no fairness or external validation is established. Correlations, source labels and model probabilities are not a medical diagnosis or a forecast of future disease. EDA of the full dataset is descriptive; fixed model settings and CV selection do not depend on test metrics.

Attribution: Janosi, Steinbrunn, Pfisterer & Detrano (1989), Heart Disease, UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X (CC BY 4.0).
