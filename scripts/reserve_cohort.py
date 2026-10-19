"""Download the future confirmation cohort without parsing or evaluating it."""

import json

from heart_disease.reservation import reserve_cohort

if __name__ == "__main__":
    print(json.dumps(reserve_cohort(), indent=2))
