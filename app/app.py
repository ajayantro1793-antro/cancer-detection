"""
app.py  —  PulmoSence · AI Lung Cancer Detection · Streamlit Application
─────────────────────────────────────────────────────────────────────────────
Features:
  - 4 switchable themes (PulmoSence / Clinical White / Midnight Blue / Deep Purple)
  - Patient details form (name, age, gender, ID, physician, notes)
  - EfficientNet-B4 classification + YOLOv8-m nodule detection
  - Grad-CAM explainability heatmap
  - PDF report with full patient details + all visualisations
  - Responsive 2-column layout

Run from project root:
    streamlit run app/app.py
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import yaml
import streamlit as st
import gdown
import os

# Download classifier model
if not os.path.exists("models/classifier/best_model.pth"):
    os.makedirs("models/classifier", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1Sq4bxKixjMun5o0FFYBBsGzRsSToLESv",
        "models/classifier/best_model.pth",
        quiet=False
    )

# Download detector model
if not os.path.exists("models/detector/runs/weights/best.pt"):
    os.makedirs("models/detector/runs/weights", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1W_5ce3GhgElAgz4N5V-XeOD9BwWFDPkX",
        "models/detector/runs/weights/best.pt",
        quiet=False
    )
import sys
import yaml
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import date

# ── Make project root importable ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from inference import load_classifier, load_detector, predict, DEVICE
from report_generator import generate_report
from themes import THEMES, apply_theme, render_theme_picker

# ── Load config ────────────────────────────────────────────────────────────
config_path = ROOT / "configs" / "config.yaml"
if config_path.exists():
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    P = cfg.get("paths", {})
    A = cfg.get("app", {})
else:
    P, A = {}, {}

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title            = "PulmoSence · AI Lung Cancer Detection",
    page_icon             = "🫁",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Apply active theme CSS ─────────────────────────────────────────────────
# This injects all :root CSS variables + component overrides for the
# currently selected theme. Must come right after set_page_config().
apply_theme()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    classifier_path = ROOT / "models" / "classifier" / "best_model.pth"
    detector_path   = ROOT / "models" / "detector" / "runs" / "weights" / "best.pt"
    alt_detector    = ROOT / "saved_models" / "yolov8_nodule_best.pt"

    if not detector_path.exists() and alt_detector.exists():
        detector_path = alt_detector

    if not classifier_path.exists():
        st.error(
            f"Classifier weights not found at `{classifier_path}`\n\n"
            "Run: `python models/classifier/train_classifier.py`"
        )
        st.stop()

    if not detector_path.exists():
        st.error(
            f"Detector weights not found at `{detector_path}`\n\n"
            "Run: `python models/detector/train_detector.py`"
        )
        st.stop()

    return load_classifier(str(classifier_path)), load_detector(str(detector_path))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    """Sidebar: logo, theme picker, device badge, confidence slider, pipeline."""
    with st.sidebar:
        t = THEMES[st.session_state.get("theme", "PulmoSence")]

        # ── Logo ──────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center; padding:20px 0 16px;">
            <div style="font-family:{t['font_display']}; font-size:26px; font-weight:700;
                        color:var(--text); letter-spacing:-0.5px;">
                Pulmo<span style="color:var(--logo-color);">Sence</span>
            </div>
            <div style="font-family:monospace; font-size:8px; letter-spacing:3px;
                        color:var(--text-muted); margin-top:4px;">
                PULMONARY AI INTELLIGENCE
            </div>
            <div style="font-family:monospace; font-size:8px; letter-spacing:2px;
                        color:var(--accent); opacity:0.3; margin-top:6px;
                        border:1px solid var(--border); display:inline-block;
                        padding:2px 10px; border-radius:8px;">
                NEXATHON 2.0 · C02
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Theme picker ──────────────────────────────────────────────────
        render_theme_picker()

        st.divider()

        # ── Device badge ──────────────────────────────────────────────────
        device_str   = DEVICE.type.upper()
        device_color = "var(--accent)" if device_str == "CUDA" else "#f0b429"
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;
                    padding:10px 14px; background:var(--accent-dim);
                    border:1px solid var(--border); border-radius:10px;">
            <div style="width:8px; height:8px; border-radius:50%; background:{device_color};
                        box-shadow:0 0 8px {device_color};"></div>
            <span style="font-family:monospace; font-size:10px; letter-spacing:1px;
                         color:var(--text-muted);">DEVICE: {device_str}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence threshold ───────────────────────────────────────────
        st.markdown("### Detection Threshold")
        conf_threshold = st.slider(
            "conf", 0.10, 0.90, 0.25, 0.05,
            label_visibility="collapsed",
            help="Minimum confidence for YOLOv8 to report a detected nodule.",
        )
        st.markdown(f"""
        <div style="text-align:right; font-family:monospace; font-size:10px;
                    color:var(--accent); opacity:0.55;
                    margin-top:-10px; margin-bottom:12px;">
            {conf_threshold:.0%} threshold
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Pipeline steps ────────────────────────────────────────────────
        st.markdown("### AI Pipeline")
        for n, label, color in [
            ("01", "HU windowing + lung mask",   "var(--accent)"),
            ("02", "EfficientNet-B4 classifier", "rgba(56,180,230,0.7)"),
            ("03", "YOLOv8-m nodule detector",   "rgba(120,140,255,0.7)"),
            ("04", "Grad-CAM heatmap",           "rgba(200,120,255,0.7)"),
            ("05", "PDF report generation",      "rgba(255,180,60,0.7)"),
        ]:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <span style="font-family:monospace; font-size:9px; color:{color}; width:16px;">{n}</span>
                <div style="width:1px; height:12px; background:rgba(255,255,255,0.06);"></div>
                <span style="font-size:11px; color:var(--text-muted); font-family:monospace;">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <p style="font-size:10px; color:var(--text-muted); opacity:0.5;
                  font-family:monospace; text-align:center; line-height:1.6;">
            For research use only.<br>Not a certified medical device.
        </p>
        """, unsafe_allow_html=True)

    return conf_threshold


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_hero():
    t = THEMES[st.session_state.get("theme", "PulmoSence")]
    st.markdown(f"""
    <div style="padding:32px 0 24px; border-bottom:1px solid var(--divider); margin-bottom:28px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="height:1px; width:28px;
                        background:linear-gradient(90deg,transparent,var(--accent));
                        opacity:0.4;"></div>
            <span style="font-family:monospace; font-size:9px; letter-spacing:3px;
                         color:var(--accent); opacity:0.45;">
                DEEP LEARNING · PULMONARY IMAGING
            </span>
            <div style="height:1px; flex:1;
                        background:linear-gradient(90deg,var(--accent),transparent);
                        opacity:0.18;"></div>
        </div>
        <div style="font-family:{t['font_display']}; font-size:clamp(28px,4vw,52px);
                    font-weight:700; line-height:1.1; color:var(--text);">
            Lung Cancer
            <span style="color:var(--accent);">Early Detection</span>
        </div>
        <div style="font-family:monospace; font-size:8px; letter-spacing:3px;
                    color:var(--accent); opacity:0.3; margin-top:6px;">
            PulmoSENCE · PULMONARY AI INTELLIGENCE
        </div>
        <p style="color:var(--text-muted); font-size:14px; line-height:1.8;
                  max-width:540px; margin:12px 0 0; font-family:{t['font_body']};">
            Fill patient details, upload a CT scan slice — PulmoSence classifies,
            detects nodules, and maps its decision via Grad-CAM.
            <strong style="color:var(--accent); opacity:0.8;">All in under 5 seconds.</strong>
        </p>
    </div>

    <div class="stat-grid">
        <div class="stat-chip">
            <div class="stat-val">94%+</div>
            <div class="stat-lbl">Accuracy</div>
            <div class="stat-sub">EfficientNet-B4</div>
        </div>
        <div class="stat-chip">
            <div class="stat-val">3mm</div>
            <div class="stat-lbl">Min Nodule</div>
            <div class="stat-sub">YOLOv8 detect</div>
        </div>
        <div class="stat-chip">
            <div class="stat-val">88%+</div>
            <div class="stat-lbl">mAP@0.5</div>
            <div class="stat-sub">nodule detection</div>
        </div>
        <div class="stat-chip">
            <div class="stat-val">&lt;5s</div>
            <div class="stat-lbl">Inference</div>
            <div class="stat-sub">full pipeline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT FORM
# ══════════════════════════════════════════════════════════════════════════════
def render_patient_form() -> dict:
    """Render patient info fields; return collected data as dict for PDF."""
    st.markdown('<div class="ps-label">PATIENT INFORMATION</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        name   = st.text_input("PATIENT NAME",     placeholder="e.g. Rajesh Kumar",      key="pt_name")
        age    = st.number_input("AGE (years)",      min_value=1, max_value=120, value=45, key="pt_age")
        pid    = st.text_input("PATIENT ID / MRN", placeholder="e.g. MRN-2026-00142",    key="pt_id")
    with c2:
        gender = st.selectbox("GENDER", ["Male", "Female", "Other", "Prefer not to say"], key="pt_gender")
        sdate  = st.date_input("SCAN DATE", value=date.today(),                            key="pt_date")
        doctor = st.text_input("REFERRING PHYSICIAN", placeholder="e.g. Dr. Priya Sharma",key="pt_doc")

    notes = st.text_area(
        "CLINICAL NOTES (optional)",
        placeholder="e.g. Persistent cough and weight loss for 3 months...",
        height=76, key="pt_notes",
    )

    return {
        "name":                name   or "Not provided",
        "age":                 str(age),
        "gender":              gender,
        "patient_id":          pid    or "Not provided",
        "scan_date":           str(sdate),
        "referring_physician": doctor or "Not provided",
        "clinical_notes":      notes  or "None",
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def render_results(result: dict):
    """Render diagnosis banner, metrics, tabs, visualisations, disclaimer."""
    risk_cfg = {
        "CRITICAL": {"color": "#ff3b5c", "bg": "rgba(255,59,92,0.08)",  "border": "rgba(255,59,92,0.3)",  "label": "CRITICAL RISK"},
        "HIGH":     {"color": "#ff7c2a", "bg": "rgba(255,124,42,0.07)", "border": "rgba(255,124,42,0.28)","label": "HIGH RISK"},
        "MEDIUM":   {"color": "#f0b429", "bg": "rgba(240,180,41,0.07)", "border": "rgba(240,180,41,0.25)","label": "MODERATE"},
        "LOW":      {"color": "#00e5b0", "bg": "rgba(0,229,176,0.06)",  "border": "rgba(0,229,176,0.25)", "label": "ALL CLEAR"},
    }
    rc  = risk_cfg.get(result["risk_level"], risk_cfg["LOW"])
    col = rc["color"]

    t = THEMES[st.session_state.get("theme", "PulmoSence")]

    # ── Diagnosis banner ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{rc['bg']}; border:1px solid {rc['border']};
                border-radius:14px; padding:22px 26px; margin-bottom:16px;
                box-shadow:0 0 50px {col}14;">
        <div style="display:flex; justify-content:space-between;
                    align-items:flex-start; gap:16px;">
            <div>
                <div style="font-family:monospace; font-size:9px; letter-spacing:2px;
                            color:{col}; opacity:0.65; margin-bottom:8px;">
                    PulmoSENCE · AI DIAGNOSIS
                </div>
                <div style="font-family:{t['font_display']}; font-size:40px;
                            font-weight:700; color:{col}; line-height:1;">
                    {result['prediction']}
                </div>
                <div style="display:inline-block; margin-top:10px; font-family:monospace;
                            font-size:10px; font-weight:700; letter-spacing:2px;
                            padding:5px 14px; border-radius:20px;
                            background:{col}18; border:1px solid {col}40; color:{col};">
                    {rc['label']}
                </div>
            </div>
            <div style="text-align:right; flex-shrink:0;">
                <div style="font-family:monospace; font-size:9px; letter-spacing:2px;
                            color:{col}; opacity:0.55; margin-bottom:4px;">CONFIDENCE</div>
                <div style="font-family:{t['font_display']}; font-size:40px;
                            font-weight:700; color:{col}; line-height:1;">
                    {result['confidence']:.1%}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ────────────────────────────────────────────────────────
    largest = max(result["nodule_sizes_mm"]) if result["nodule_sizes_mm"] else 0
    stage   = "Stage I-II" if result["prediction"] == "Malignant" else result["prediction"]
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <p class="metric-val" style="color:{col};">{result['prediction']}</p>
            <p class="metric-lbl">DIAGNOSIS</p>
        </div>
        <div class="metric-card">
            <p class="metric-val" style="color:{col};">{result['confidence']:.1%}</p>
            <p class="metric-lbl">CONFIDENCE</p>
        </div>
        <div class="metric-card">
            <p class="metric-val" style="color:{col};">{result['nodule_count']}</p>
            <p class="metric-lbl">NODULES</p>
        </div>
        <div class="metric-card">
            <p class="metric-val" style="color:{col};">{largest} mm</p>
            <p class="metric-lbl">LARGEST</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["OVERVIEW", "PROBABILITIES", "NODULES"])

    with tab1:
        st.markdown(f"""
        <div style="padding:14px; border-radius:10px;
                    background:var(--accent-dim); border:1px solid var(--border);
                    margin-top:8px; font-family:monospace; font-size:11px;
                    line-height:1.7; color:var(--text-muted);">
            Grad-CAM heatmap active — warm (red) regions indicate EfficientNet-B4 attention zones.<br>
            Estimated stage: <strong style="color:{col};">{stage}</strong>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        prob_colors = {"Normal": "#00e5b0", "Benign": "#f0b429", "Malignant": "#ff3b5c"}
        for cls, prob in result["probabilities"].items():
            pc = prob_colors.get(cls, "#00e5b0")
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-header">
                    <span style="color:var(--text);">{cls}</span>
                    <span style="color:{pc}; font-weight:700;">{prob:.1%}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{prob*100:.1f}%;
                         background:linear-gradient(90deg,{pc}60,{pc});
                         box-shadow:0 0 8px {pc}50;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        if result["nodule_sizes_mm"]:
            for i, (sz, sc) in enumerate(
                    zip(result["nodule_sizes_mm"], result["scores"]), start=1):
                dc  = "#ff3b5c" if sz >= 6 else "#f0b429"
                sig = (
                    '<span style="font-size:9px;color:#fca5a5;'
                    'background:rgba(255,59,92,0.1);border:1px solid rgba(255,59,92,0.25);'
                    'padding:2px 6px;border-radius:4px;margin-left:8px;">'
                    '>=6mm SIGNIFICANT</span>'
                ) if sz >= 6 else ""
                st.markdown(f"""
                <div class="nodule-row">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div style="width:10px;height:10px;border-radius:50%;
                                    background:{dc};box-shadow:0 0 8px {dc}80;
                                    flex-shrink:0;"></div>
                        <span style="color:var(--text);">Nodule {i}</span>{sig}
                    </div>
                    <div style="display:flex;gap:20px;">
                        <span style="color:var(--text);font-weight:700;">{sz} mm</span>
                        <span style="color:var(--text-muted);">{sc:.0%} conf</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:28px 0;">
                <p style="color:var(--accent);font-family:monospace;
                          font-size:12px;margin:0;">No nodules detected</p>
                <p style="color:var(--text-muted);font-size:10px;
                          font-family:monospace;margin:4px 0 0;">
                    above confidence threshold
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Visualisation panels ───────────────────────────────────────────────
    st.markdown('<div class="ps-label" style="margin-top:24px;">CT SCAN VISUALISATIONS</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(result["original_image"],
                 caption="ORIGINAL CT SLICE", use_container_width=True)
    with c2:
        st.image(result["detection_image"],
                 caption=f"YOLOV8 · {result['nodule_count']} NODULE(S)",
                 use_container_width=True)
    with c3:
        st.image(result["heatmap_image"],
                 caption="GRAD-CAM · RED = HIGH INFLUENCE",
                 use_container_width=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.warning(
        "**Clinical Disclaimer:** AI screening aid only. Not a certified medical device. "
        "All findings must be reviewed by a licensed radiologist before any clinical action."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Load models
    classifier, detector = load_models()

    # Sidebar (also runs render_theme_picker inside)
    conf_threshold = render_sidebar()

    # Re-apply theme after sidebar interaction (theme may have just changed)
    apply_theme()

    # Hero section
    render_hero()

    # ── Two column layout ─────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    # ═══════════ LEFT COLUMN ═══════════
    with left_col:

        # Patient info card
        st.markdown('<div class="ps-card">', unsafe_allow_html=True)
        patient_info = render_patient_form()
        st.markdown('</div>', unsafe_allow_html=True)

        # CT scan upload card
        st.markdown('<div class="ps-card">', unsafe_allow_html=True)
        st.markdown('<div class="ps-label">CT SCAN INPUT</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop CT scan here — PNG / JPG / 512x512 recommended",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="LOADED · click to change",
                     use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Analyse button
        if uploaded_file:
            if st.button("ANALYZE SCAN", type="primary", use_container_width=True):
                with st.spinner("Running AI analysis pipeline..."):
                    try:
                        result = predict(
                            image_input    = pil_image,
                            classifier     = classifier,
                            detector       = detector,
                            conf_threshold = conf_threshold,
                        )
                        result["patient_info"] = patient_info
                        st.session_state["result"]    = result
                        st.session_state["pil_image"] = pil_image
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.exception(e)

        # Reset button
        if "result" in st.session_state:
            if st.button("UPLOAD NEW SCAN", type="secondary", use_container_width=True):
                for k in ["result", "pil_image"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ═══════════ RIGHT COLUMN ═══════════
    with right_col:
        if "result" in st.session_state:
            result = st.session_state["result"]

            render_results(result)

            st.divider()

            # PDF download
            st.markdown('<div class="ps-label">DOWNLOAD REPORT</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_report(
                    result,
                    patient_info=result.get("patient_info", {}),
                )
            fname = (
                f"PulmoSence_"
                f"{result.get('patient_info',{}).get('name','patient').replace(' ','_')}_"
                f"{result.get('patient_info',{}).get('scan_date','report')}.pdf"
            )
            st.download_button(
                label     = "DOWNLOAD PDF DIAGNOSTIC REPORT",
                data      = pdf_bytes,
                file_name = fname,
                mime      = "application/pdf",
                use_container_width=True,
            )

        else:
            # Empty state placeholder
            st.markdown("""
            <div style="min-height:520px; border-radius:18px;
                        border:1.5px dashed var(--border);
                        background:var(--accent-dim);
                        display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        gap:14px; text-align:center; padding:48px;">
                <div style="font-size:52px; opacity:0.3;">🫁</div>
                <p style="margin:0; color:var(--text-muted);
                          font-family:var(--font-display); font-size:13px;">
                    Results appear here
                </p>
                <p style="margin:0; color:var(--accent); opacity:0.25;
                          font-size:11px; font-family:monospace;">
                    Fill patient details &bull; Upload CT scan &bull; Click Analyze
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <p style="text-align:center; font-family:monospace; font-size:9px; letter-spacing:2px;
              color:var(--accent); opacity:0.15; margin-top:6px;">
        PulmoSENCE · PULMONARY AI INTELLIGENCE · NEXATHON 2.0 C02 · FOR RESEARCH USE ONLY
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

