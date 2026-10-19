# Why does recall fall on the Hungarian cohort?

Descriptive investigation of frozen run `27586d03470741438f644f6067b70f22`. No model, probability method or threshold was changed. No future reserved cohort was read.

The existing Hungarian check reports recall 0.4906, specificity 0.9679, and ROC-AUC 0.8987. At the fixed 0.5 threshold the model misses 54 of 106 presence-labelled observations. Ranking performance and threshold sensitivity describe different properties.

## Observed availability shift

| Feature | Cleveland training missing | Hungarian missing |
|---|---:|---:|
| ca | 0.4% | 99.0% |
| thal | 0.4% | 90.4% |
| slope | 0.0% | 64.5% |
| chol | 0.0% | 7.5% |
| fbs | 0.0% | 2.7% |
| trestbps | 0.0% | 0.3% |
| restecg | 0.0% | 0.3% |
| thalach | 0.0% | 0.3% |
| exang | 0.0% | 0.3% |
| age | 0.0% | 0.0% |
| sex | 0.0% | 0.0% |
| cp | 0.0% | 0.0% |
| oldpeak | 0.0% | 0.0% |

The leading importance features include vessel count and chest-pain type. Missing vessel/thallium/slope measurements are replaced by Cleveland-trained imputation values. A site with a different availability pattern can therefore be represented poorly even when its outcome ranking appears useful.

## Controlled training-only stress check

5-fold stratified out-of-fold predictions on the 242 original training observations, seed 42. Every fit uses only that fold's training subset. The fitted model is unchanged between a fold's validation conditions. Only validation feature availability is deliberately perturbed.

| Validation condition | Recall | Specificity | ROC-AUC |
|---|---:|---:|---:|
| Unmodified validation records | 0.7838 | 0.8321 | 0.8980 |
| Mask ca and thal in validation | 0.2793 | 0.9542 | 0.8185 |
| Mask ca, thal and slope in validation | 0.2162 | 1.0000 | 0.8085 |

Stress checks mask predictors on genuine training-fold validation observations. They are controlled perturbations, not a new natural cohort or fabricated dataset. Changes show dependence on feature availability, not proof that missingness alone caused the Hungarian recall gap. Site selection, class prevalence, covariate differences and missingness mechanisms remain confounded. Observed external slices may be tiny and must not select models or thresholds.

External missing/observed slices with sample counts and Wilson recall intervals are in transfer_observed_slices.csv. Their class composition and small observed groups prevent a simple causal comparison. No attempt was made to fill missing Hungarian measurements with invented observations or to tune against its already-viewed outcomes.

## Next step

See docs/next_experiment_protocol.md. Future candidates and selection criteria are declared using training data and observed measurement availability. A different UCI cohort is reserved without opening its labels or evaluating a model. The current deployed model remains the verified original run.
