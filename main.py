import argparse
import json
import logging
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Resume Screening & Ranking System")
    parser.add_argument("--input", required=True, help="Folder containing resumes")
    parser.add_argument("--output", required=True, help="Output results.json path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    results = run_pipeline(Path(args.input))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if results["batch_summary"]["interrupted"]:
        logging.warning("Partial results written to %s", output_path)
    else:
        logging.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
