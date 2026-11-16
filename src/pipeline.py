"""
End-to-End Solar UAV Inspection Pipeline.
Coordinates preprocessing, panel segmentation, anomaly detection,
visualization, report generation, and performance evaluation.
"""

import os
import glob
from typing import Dict, List, Any, Optional
import cv2

from .preprocessing import ImagePreprocessor, PanelSegmenter
from .detection import AnomalyDetector
from .visualization import Visualizer
from .reporting import ReportGenerator
from .evaluation import Evaluator


class SolarInspectionPipeline:
    """Orchestrates the entire UAV drone inspection and anomaly detection lifecycle."""

    def __init__(self, clahe_clip: float = 3.0, hotspot_thresh: float = 6.0):
        self.clahe_clip = clahe_clip
        self.preprocessor = ImagePreprocessor()
        self.segmenter = PanelSegmenter()
        self.detector = AnomalyDetector(hotspot_delta_thresh=hotspot_thresh)
        self.visualizer = Visualizer()
        self.reporter = ReportGenerator()
        self.evaluator = Evaluator()

    def process_image(
        self,
        image_path: str,
        output_annotated_dir: str = "outputs/annotated",
        output_processed_dir: str = "data/processed"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing on a single UAV drone aerial frame.
        """
        filename = os.path.basename(image_path)
        raw_bgr = self.preprocessor.load_image(image_path)
        modality = self.preprocessor.detect_modality(raw_bgr, filename=filename)

        # Preprocessing & Contrast Enhancement
        enhanced_bgr = self.preprocessor.apply_clahe(raw_bgr, clip_limit=self.clahe_clip)

        if output_processed_dir:
            os.makedirs(output_processed_dir, exist_ok=True)
            proc_path = os.path.join(output_processed_dir, f"proc_{filename}")
            cv2.imwrite(proc_path, enhanced_bgr)

        # Panel Segmentation
        panels, mask = self.segmenter.segment_panels(enhanced_bgr)

        # Anomaly Detection
        defects = self.detector.analyze_panels(panels, modality=modality)
        for d in defects:
            d["image_name"] = filename

        # Visualization
        annotated_bgr = self.visualizer.draw_annotations(enhanced_bgr, panels, defects)

        if output_annotated_dir:
            os.makedirs(output_annotated_dir, exist_ok=True)
            ann_path = os.path.join(output_annotated_dir, f"annotated_{filename}")
            cv2.imwrite(ann_path, annotated_bgr)

        return {
            "image_name": filename,
            "modality": modality,
            "raw_image": raw_bgr,
            "enhanced_image": enhanced_bgr,
            "annotated_image": annotated_bgr,
            "panels": panels,
            "defects": defects
        }

    def run_batch(
        self,
        input_dir: str = "data/original",
        output_dir: str = "outputs",
        ground_truth_path: Optional[str] = "data/ground_truth/annotations.csv",
        generate_pdf: bool = True,
        generate_html: bool = True
    ) -> Dict[str, Any]:
        """
        Executes batch inspection across all aerial images in the input directory.
        Outputs annotated images, inspection_results.csv, and summary reports.
        """
        annotated_dir = os.path.join(output_dir, "annotated")
        reports_dir = os.path.join(output_dir, "reports")
        processed_dir = os.path.join("data", "processed")

        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))

        if not image_files:
            raise FileNotFoundError(f"No image files found in input directory: {input_dir}")

        total_panels_all = []
        total_defects_all = []
        batch_results = []

        print(f"\n=======================================================")
        print(f"  SOLAR UAV INSPECTION & ANOMALY DETECTION PIPELINE")
        print(f"=======================================================")
        print(f"Found {len(image_files)} aerial drone survey frames in '{input_dir}'\n")

        for idx, img_path in enumerate(image_files, start=1):
            fname = os.path.basename(img_path)
            print(f"[{idx}/{len(image_files)}] Processing {fname}...")
            res = self.process_image(img_path, output_annotated_dir=annotated_dir, output_processed_dir=processed_dir)
            batch_results.append(res)
            total_panels_all.extend(res["panels"])
            total_defects_all.extend(res["defects"])

            print(f"       -> Modality: {res['modality']}")
            print(f"       -> Identified Panels: {len(res['panels'])}")
            print(f"       -> Detected Anomalies: {len(res['defects'])}")
            for d in res["defects"]:
                print(f"          * [{d['severity']}] {d['type_name']} on {d['panel_id']} (Conf: {d['confidence']*100:.0f}%)")

        # Compile Summary & Export Reports
        summary = self.reporter.calculate_summary(total_panels_all, total_defects_all)

        csv_path = os.path.join(reports_dir, "inspection_results.csv")
        self.reporter.export_csv(csv_path, total_defects_all)
        print(f"\n[+] Exported CSV Defect Log: {csv_path}")

        if generate_html:
            html_path = os.path.join(reports_dir, "inspection_report.html")
            self.reporter.export_html(html_path, summary, total_defects_all)
            print(f"[+] Exported HTML Audit Report: {html_path}")

        if generate_pdf:
            pdf_path = os.path.join(reports_dir, "inspection_report.pdf")
            self.reporter.export_pdf(pdf_path, summary, total_defects_all)
            print(f"[+] Exported PDF Engineering Audit: {pdf_path}")

        # Benchmark against Ground Truth if available
        eval_metrics = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            gt_records = self.evaluator.load_ground_truth(ground_truth_path)
            eval_metrics = self.evaluator.evaluate(total_defects_all, gt_records)
            print(f"\n=======================================================")
            print(f"  MODEL BENCHMARK EVALUATION (GROUND TRUTH)")
            print(f"=======================================================")
            print(f"Precision: {eval_metrics['precision'] * 100:.1f}%")
            print(f"Recall:    {eval_metrics['recall'] * 100:.1f}%")
            print(f"F1-Score:  {eval_metrics['f1_score'] * 100:.1f}%")
            print(f"TP: {eval_metrics['true_positives']} | FP: {eval_metrics['false_positives']} | FN: {eval_metrics['false_negatives']}")

        print(f"\n=======================================================")
        print(f"  INSPECTION SUMMARY")
        print(f"=======================================================")
        print(f"Total Modules Surveyed:   {summary['total_panels']}")
        print(f"Fleet Health Index:       {summary['health_index_pct']}%")
        print(f"Defective Modules:        {summary['defective_panels']}")
        print(f"Critical Deficiencies:    {summary['critical_defects']}")
        print(f"Estimated Capacity Loss:  {summary['total_power_loss_kw']} kW")
        print(f"Estimated Annual Loss:    ${summary['annual_financial_loss_usd']}\n")

        return {
            "summary": summary,
            "defects": total_defects_all,
            "evaluation": eval_metrics
        }
