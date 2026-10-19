"""Smoke-test inference using one real source record; never used as evaluation."""
from _common import data_parser, load_args
from heart_disease.config import FEATURES, MODEL_PATH
from heart_disease.prediction import load_artifact, predict_records
from pathlib import Path


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    try:
        pipeline, metadata = load_artifact(args.model)
        dataset = load_args(args)
        record = dataset.frame[FEATURES].dropna().iloc[[0]]
        result = predict_records(pipeline, record)
    except (ValueError, RuntimeError, OSError, IndexError) as exc:
        parser.exit(1, f"Prediction smoke test failed: {exc}\n")
    print(f"Model: {metadata['selected_model']}; one real complete source record (functional check only)")
    print(result.to_string(index=False))
    print("Model output is educational, uncalibrated and not a medical diagnosis.")


if __name__ == "__main__":
    main()
