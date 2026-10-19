"""Small artificial records for isolated unit tests ONLY; never experiment data."""
import numpy as np
import pandas as pd
import pytest

from heart_disease.config import FEATURES
from heart_disease.data import load_dataset


@pytest.fixture
def frame():
    rng = np.random.default_rng(123)
    n = 40
    values = {
        "age": np.arange(30, 30 + n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(140, 350, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 190, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    }
    return pd.DataFrame(values, columns=FEATURES).assign(num=np.arange(n) % 2)


@pytest.fixture
def dataset(frame, tmp_path):
    path = tmp_path / "unit_fixture.csv"
    frame.to_csv(path, index=False)
    return load_dataset(path)
