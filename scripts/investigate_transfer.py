"""Investigate the existing transfer gap without changing the deployed pipeline."""

from heart_disease.config import MODEL_PATH, REPORT_DIR
from heart_disease.transfer import investigate

if __name__ == "__main__":
    result = investigate(MODEL_PATH, REPORT_DIR)
    for row in result["training_only_stress"]:
        print(row)
    print(
        f"Investigation saved to {REPORT_DIR / 'transfer_diagnostics.md'}; deployment model unchanged."
    )
