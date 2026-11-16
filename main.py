"""
Main Entry Point for Solar UAV Inspection & Anomaly Detection System.

Usage:
    python main.py                              # Runs full pipeline on data/original
    python main.py --input path/to/images       # Runs pipeline on custom drone directory
    python main.py --evaluate                   # Computes Precision/Recall vs ground truth
    python main.py --web                        # Launches interactive Web Inspection GUI
"""

import argparse
import sys
import os

from src.pipeline import SolarInspectionPipeline


def main():
    parser = argparse.ArgumentParser(description="Solar UAV Drone Inspection & Anomaly Detection System")
    parser.add_argument("--input", "-i", type=str, default="data/original", help="Input directory of drone aerial images")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory for annotated imagery and reports")
    parser.add_argument("--ground-truth", "-g", type=str, default="data/ground_truth/annotations.csv", help="Path to annotations CSV for evaluation")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report generation")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation against ground truth")
    parser.add_argument("--web", action="store_true", help="Launch interactive Streamlit web dashboard")

    args = parser.parse_args()

    # Launch Web Interface if requested
    if args.web:
        print("Launching Solar UAV Inspection Interactive Web Dashboard...")
        standalone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "web_app.py")
        if not os.path.exists(standalone_path):
            # Fallback to streamlit running the standalone app
            os.system(f"python -m streamlit run {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'pipeline.py')}")
        else:
            os.system(f"python -m streamlit run {standalone_path}")
        return

    # Run End-to-End Pipeline
    pipeline = SolarInspectionPipeline()
    gt_path = args.ground_truth if (args.evaluate or os.path.exists(args.ground_truth)) else None

    results = pipeline.run_batch(
        input_dir=args.input,
        output_dir=args.output,
        ground_truth_path=gt_path,
        generate_pdf=not args.no_pdf,
        generate_html=not args.no_html
    )

    print("Pipeline execution complete. Check outputs/ for results.")


if __name__ == "__main__":
    main()
