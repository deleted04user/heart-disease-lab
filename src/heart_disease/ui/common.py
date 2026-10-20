from pathlib import Path

import streamlit as st

from heart_disease import config as settings
from heart_disease.io import read_json


def json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (ValueError, OSError) as exc:
        st.warning(
            f"Cannot read {path.name}: {exc}. Recreate reports with python scripts/train.py."
        )
        return None


def figure(name: str, caption: str = ""):
    path = settings.REPORT_DIR / "figures" / name
    if path.exists():
        try:
            st.image(str(path), caption=caption or None, width="stretch")
        except Exception:
            st.info(
                f"Figure {name} cannot be displayed. Regenerate it with python scripts/train.py."
            )
    else:
        st.info(f"Figure {name} is unavailable. Run python scripts/train.py.")


def header(kicker: str, title: str, subtitle: str):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(subtitle)


def recover(message: str):
    st.info(message)
    st.code("python scripts/download_data.py\npython scripts/train.py", language="powershell")


def compatible_report(report: dict | None, metadata: dict | None) -> bool:
    if not report or not {
        "run_id",
        "selected_model",
        "test_results",
        "configuration",
        "data",
    }.issubset(report):
        recover("No complete evaluation report is available yet.")
        return False
    if metadata and report.get("run_id") != metadata.get("run_id"):
        st.error(
            "Model and report belong to different runs. Rerun training with matching model/report paths."
        )
        return False
    return True
