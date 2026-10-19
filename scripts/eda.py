"""Generate descriptive reports and figures from one real validated source."""

from _common import configure_logging, data_parser, load_args
from heart_disease.eda import run_eda


def main():
    parser = data_parser(__doc__)
    args = parser.parse_args()
    configure_logging()
    try:
        dataset = load_args(args)
        summary = run_eda(dataset, args.reports)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"EDA failed: {exc}\n")
    print(
        f"EDA complete: {dataset.metadata['rows']} actual records; positive fraction {summary['positive_fraction']:.4f}. Reports: {args.reports.resolve()}"
    )


if __name__ == "__main__":
    main()
