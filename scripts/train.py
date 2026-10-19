"""Compare three fixed pipelines and save real evaluation and inference artifacts."""
from _common import configure_logging, data_parser, load_args
from heart_disease.config import MODEL_PATH, TEST_SIZE
from heart_disease.eda import run_eda
from heart_disease.modeling import train
from heart_disease.experiments import RUNS_DIR, complete_run, create_run, preserve_existing, publish_run
from pathlib import Path


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--external", type=Path, default=None, help="One independently sourced original-UCI-coded cohort, evaluated after selection")
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--no-activate", action="store_true", help="Save a complete versioned experiment without replacing active model/reports")
    args = parser.parse_args()
    configure_logging()
    try:
        dataset = load_args(args)
        if not args.no_activate:
            preserve_existing(args.model, args.reports, args.runs_dir)
        run_dir = create_run(args.runs_dir)
        run_reports, run_model = run_dir / "reports", run_dir / "models/heart_model.joblib"
        run_eda(dataset, run_reports)
        report = train(dataset, model_path=run_model, report_dir=run_reports, test_size=args.test_size, cv_repeats=args.cv_repeats, bootstrap_resamples=args.bootstrap_resamples, external_path=args.external)
        complete_run(run_dir, report)
        if not args.no_activate:
            publish_run(run_dir, args.model, args.reports)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    chosen = report['selected_model']
    print(f"Selected from training CV: {chosen}")
    print(f"Probability method: {report['calibration']['selected_method']}")
    print(f"Historical Cleveland audit: {report['test_results'][chosen]}")
    print(f"External status: {report['external']['status']}")
    print(f"Immutable run: {run_dir}")
    if not args.no_activate:
        print(f"Active model: {args.model.resolve()}\nActive reports: {args.reports.resolve()}")


if __name__ == "__main__":
    main()
