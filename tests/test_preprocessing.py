import numpy as np

from heart_disease.config import FEATURES, NUMERIC
from heart_disease.preprocessing import build_preprocessor


def test_imputation_scaling_and_target_exclusion(dataset):
    train = dataset.frame.iloc[:25].copy()
    train.loc[0, "chol"] = np.nan
    train.loc[1, "ca"] = np.nan
    pre = build_preprocessor()
    transformed = pre.fit_transform(train)
    assert np.isfinite(transformed).all()
    assert transformed.shape[0] == 25
    assert all("target" not in name and "num" != name for name in pre.get_feature_names_out())
    medians = pre.named_transformers_["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(medians, train[NUMERIC].median())
    np.testing.assert_allclose(transformed[:, :len(NUMERIC)].mean(axis=0), 0, atol=1e-12)
    # Transforming held-out extremes must not refit training medians/scalers.
    heldout = dataset.frame.iloc[25:][FEATURES].copy()
    heldout["chol"] = 800
    pre.transform(heldout)
    np.testing.assert_allclose(pre.named_transformers_["numeric"].named_steps["impute"].statistics_, medians)


def test_unknown_categories_safe_inside_encoder(dataset):
    pre = build_preprocessor()
    pre.fit(dataset.frame[FEATURES])
    unknown = dataset.frame[FEATURES].iloc[[0]].copy()
    unknown["cp"] = 99
    result = pre.transform(unknown)
    assert np.isfinite(result).all()
    assert result.shape[1] == len(pre.get_feature_names_out())
