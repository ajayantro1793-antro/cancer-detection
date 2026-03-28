"""
report_generator.py  —  PDF Report Generation
─────────────────────────────────────────────────────────────────────────────
Generates a professional, printable PDF diagnostic report using the fpdf2
library. The report includes:
  - Patient scan metadata and timestamp
  - Diagnosis result with confidence score and risk level
  - Probability breakdown for all 3 classes
  - Nodule count and estimated sizes
  - The three visualisation panels (original, detections, Grad-CAM)
  - A clear disclaimer that AI output requires radiologist review

The report is designed to be handed to a radiologist for verification —
it is NOT intended as a standalone clinical decision.

Fix applied:
  - Temp image files are now fully closed before fpdf reads them,
    preventing PermissionError [WinError 32] on Windows.
─────────────────────────────────────────────────────────────────────────────
"""

import io
import os
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from fpdf import FPDF


class DiagnosticReport(FPDF):
    """
    Custom FPDF subclass for the lung cancer diagnostic report.
    We override header() and footer() to add our branding and disclaimer
    to every page automatically.
    """

    def header(self):
        # Blue header bar
        self.set_fill_color(37, 99, 235)      # #2563EB
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        self.cell(0, 10, "AI-Powered Lung Cancer Detection System", align="C")
        self.ln(14)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)    # muted grey
        self.cell(0, 10,
                  "DISCLAIMER: This report is AI-generated and must be reviewed "
                  "by a qualified radiologist before any clinical decision is made.  "
                  f"Page {self.page_no()}",
                  align="C")


def generate_report(result: dict,
                    filename: str = "diagnostic_report.pdf",
                    patient_info: dict = None) -> bytes:
    """
    Generate a complete PDF diagnostic report from an inference result dict.

    Args:
        result:   The dict returned by inference.predict()
        filename: Desired output filename (used only as metadata)

    Returns:
        PDF content as bytes, suitable for st.download_button() in Streamlit.
    """
    pdf = DiagnosticReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    timestamp    = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    patient_info = patient_info or {}

    # ── Section 1: Report Header Info ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "PulmoSence - AI Lung Cancer Diagnostic Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, f"Generated: {timestamp}", ln=True)
    pdf.cell(0, 6, "Powered by EfficientNet-B4 + YOLOv8-m + Grad-CAM", ln=True)
    pdf.ln(4)

    # ── Section 1b: Patient Details Box ───────────────────────────────────
    if patient_info:
        pdf.set_fill_color(240, 249, 255)
        pdf.set_draw_color(37, 99, 235)
        box_y = pdf.get_y()
        pdf.rect(10, box_y, 190, 46, "DF")
        pdf.set_y(box_y + 3)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(0, 6, "  Patient Information", ln=True)

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(15, 23, 42)

        # Row 1: Name | Age | Gender
        name   = patient_info.get("name",   "Not provided")
        age    = patient_info.get("age",    "N/A")
        gender = patient_info.get("gender", "N/A")
        pdf.cell(65, 5.5, f"  Name: {name}", border=0)
        pdf.cell(62, 5.5, f"Age: {age} yrs", border=0)
        pdf.cell(63, 5.5, f"Gender: {gender}", border=0, ln=True)

        # Row 2: Patient ID | Scan Date | Referring Physician
        pid    = patient_info.get("patient_id",          "Not provided")
        sdate  = patient_info.get("scan_date",            "N/A")
        doctor = patient_info.get("referring_physician",  "Not provided")
        pdf.cell(65, 5.5, f"  MRN / ID: {pid}", border=0)
        pdf.cell(62, 5.5, f"Scan Date: {sdate}", border=0)
        pdf.cell(63, 5.5, f"Physician: {doctor}", border=0, ln=True)

        # Row 3: Clinical notes (truncated to 120 chars to fit)
        notes = patient_info.get("clinical_notes", "None")
        if len(notes) > 120:
            notes = notes[:117] + "..."
        pdf.cell(0, 5.5, f"  Notes: {notes}", border=0, ln=True)

        pdf.ln(5)

    # ── Section 2: Diagnosis Result Box ───────────────────────────────────
    prediction  = result["prediction"]
    confidence  = result["confidence"]
    risk_level  = result["risk_level"]

    # Choose fill colour based on risk level
    risk_colours = {
        "CRITICAL": (239, 68,  68),   # red
        "HIGH":     (249, 115, 22),   # orange
        "MEDIUM":   (245, 158, 11),   # amber
        "LOW":      (16,  185, 129),  # green
    }
    r, g, b = risk_colours.get(risk_level, (37, 99, 235))

    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.rect(10, pdf.get_y(), 190, 20, "F")
    pdf.set_y(pdf.get_y() + 4)
    pdf.cell(0, 12,
             f"Diagnosis:  {prediction}   |   Confidence: {confidence:.1%}   |   Risk: {risk_level}",
             align="C", ln=True)
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    # ── Section 3: Probability Breakdown ──────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Class Probabilities", ln=True)
    pdf.set_font("Helvetica", "", 10)

    class_colours = {
        "Normal":    (5,   150, 105),
        "Benign":    (217, 119, 6),
        "Malignant": (239, 68,  68),
    }
    for class_name, prob in result["probabilities"].items():
        cr, cg, cb = class_colours.get(class_name, (71, 85, 105))
        bar_w = int(140 * prob)
        y0 = pdf.get_y()

        # Label
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(30, 6, class_name)

        # Bar background
        pdf.set_fill_color(226, 232, 240)
        pdf.rect(45, y0 + 1, 140, 4, "F")

        # Bar fill
        pdf.set_fill_color(cr, cg, cb)
        pdf.rect(45, y0 + 1, bar_w, 4, "F")

        # Percentage
        pdf.set_xy(190, y0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(cr, cg, cb)
        pdf.cell(0, 6, f"{prob:.1%}", align="R", ln=True)

    pdf.ln(4)

    # ── Section 4: Nodule Summary ──────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Nodule Detection Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, f"Total nodules detected: {result['nodule_count']}", ln=True)

    if result["nodule_sizes_mm"]:
        sizes_str = ",  ".join([f"{s} mm" for s in result["nodule_sizes_mm"]])
        pdf.cell(0, 6, f"Estimated sizes: {sizes_str}", ln=True)
        pdf.cell(0, 6,
                 f"Largest nodule: {max(result['nodule_sizes_mm'])} mm  "
                 f"(clinically significant threshold: 6mm)",
                 ln=True)
    else:
        pdf.cell(0, 6, "No nodules detected above confidence threshold.", ln=True)

    pdf.ln(4)

    # ── Section 5: Visualisation Images ───────────────────────────────────
    # FIX: Temp files are fully closed before fpdf reads them to prevent
    # PermissionError [WinError 32] on Windows where files cannot be read
    # and deleted while still held open by another handle.
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "CT Scan Analysis Visualisations", ln=True)

    image_keys   = ["original_image", "detection_image", "heatmap_image"]
    image_labels = ["Original CT Slice", "Nodule Detections (YOLOv8)", "Grad-CAM Heatmap"]
    x_positions  = [10, 75, 140]
    img_w, img_h = 58, 58

    y_start = pdf.get_y()
    for key, label, x in zip(image_keys, image_labels, x_positions):
        img_array = result.get(key)
        if img_array is not None:
            tmp_path = None
            try:
                # Step 1: Write the image and CLOSE the file handle
                #         before fpdf tries to open/read it.
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pil_img = Image.fromarray(img_array.astype(np.uint8))
                    pil_img.save(tmp_path)
                # File is fully closed here — safe for fpdf to read on Windows

                # Step 2: fpdf reads the closed file
                pdf.image(tmp_path, x=x, y=y_start, w=img_w, h=img_h)

            finally:
                # Step 3: Safe delete — won't crash if Windows is slow to
                #         release the lock. OS cleans Temp folder anyway.
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        pass  # Windows still locking — safe to ignore

            # Label below image
            pdf.set_xy(x, y_start + img_h + 1)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(img_w, 5, label, align="C")

    pdf.set_y(y_start + img_h + 8)
    pdf.ln(4)

    # ── Section 6: Clinical Disclaimer ────────────────────────────────────
    pdf.set_fill_color(239, 246, 255)          # light blue background
    pdf.set_draw_color(37, 99, 235)
    pdf.rect(10, pdf.get_y(), 190, 26, "DF")
    pdf.set_y(pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 5, "  Important Clinical Notice", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(190, 5,
        "  This AI-generated report is a screening aid only. It does not constitute "
        "a medical diagnosis. All findings must be reviewed and confirmed by a licensed "
        "radiologist or oncologist before any clinical action is taken. The AI system "
        "has a target accuracy of 94% - 6% of cases may be misclassified.",
        align="L")

    # Return PDF as raw bytes
    return bytes(pdf.output())

