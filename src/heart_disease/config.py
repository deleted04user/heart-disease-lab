"""Shared schema, paths, labels, and experiment defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("HEART_DATA_PATH", ROOT / "data/raw/processed.cleveland.data"))
MODEL_PATH = Path(os.environ.get("HEART_MODEL_PATH", ROOT / "models/heart_model.joblib"))
REPORT_DIR = Path(os.environ.get("HEART_REPORT_DIR", ROOT / "reports"))
SEED = 42
TEST_SIZE = 0.2
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
UCI_ARCHIVE = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/45/heart+disease"
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"
# Broad data-entry bounds, NOT medical reference intervals. Zeros in cholesterol
# or blood pressure in some source variants are not accepted as measurements.
BOUNDS = {"age": (1, 120), "trestbps": (50, 300), "chol": (50, 800), "thalach": (20, 250), "oldpeak": (-5.0, 10.0)}
CATEGORY_LABELS = {
    "sex": {0: "Female (source code 0)", 1: "Male (source code 1)"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "At or below 120 mg/dL", 1: "Above 120 mg/dL"},
    "restecg": {0: "Normal", 1: "ST–T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
FIELD_LABELS = {
    "age": "Age (years)", "sex": "Sex recorded in source", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)", "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar", "restecg": "Resting ECG", "thalach": "Maximum heart rate (bpm)",
    "exang": "Exercise-induced angina", "oldpeak": "ST depression relative to rest",
    "slope": "Peak exercise ST slope", "ca": "Major vessels seen by fluoroscopy", "thal": "Thallium test result",
}
DISCLAIMER = "Educational research demonstration. Not clinically validated. Do not use this output to diagnose, treat, or make medical decisions."
