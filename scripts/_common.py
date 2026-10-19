"""Shared CLI options and source-tree imports for direct script execution."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_disease.config import DATA_PATH, REPORT_DIR
from heart_disease.data import load_dataset
from heart_disease.logging_config import configure_logging


def data_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="One source file; original UCI codes by default")
    parser.add_argument("--schema", choices=["uci", "kaggle"], default="uci")
    parser.add_argument("--target-semantics", choices=["uci", "binary"], default=None, help="Required for target/HeartDisease headers; assert positive labels mean disease presence")
    parser.add_argument("--site", default="Cleveland", help="One exact Kaggle dataset value; sites are never combined")
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    return parser


def load_args(args):
    return load_dataset(args.data, schema=args.schema, target_semantics=args.target_semantics, site=args.site)


