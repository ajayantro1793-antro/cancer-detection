"""
themes.py  —  PulmoSence Multi-Theme System
─────────────────────────────────────────────────────────────────────────────
4 themes:
  1. PulmoSence     — deep-space bioluminescent dark  (default)
  2. Clinical White — clean bright medical dashboard
  3. Midnight Blue  — navy clinical dark
  4. Deep Purple    — radiology dark violet

Usage in app.py:
    from themes import THEMES, apply_theme, render_theme_picker

    # inside render_sidebar():
    render_theme_picker()

    # at top of main(), after st.set_page_config():
    apply_theme()
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# THEME DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
THEMES = {

    # ── 1. PulmoSence — bioluminescent deep space ─────────────────────────
    "PulmoSence": {
        "bg":           "#00060f",
        "surface":      "#010d1e",
        "card":         "#021122",
        "border":       "rgba(0,229,176,0.12)",
        "border_h":     "rgba(0,229,176,0.32)",
        "accent":       "#00e5b0",
        "accent_dim":   "rgba(0,229,176,0.08)",
        "text":         "#d4f5e8",
        "text_muted":   "rgba(160,220,200,0.45)",
        "input_bg":     "rgba(0,229,176,0.04)",
        "input_bd":     "rgba(0,229,176,0.15)",
        "input_focus":  "rgba(0,229,176,0.4)",
        "btn_fg":       "#001a12",
        "btn_g1":       "#00b386",
        "btn_g2":       "#00e5b0",
        "btn_g3":       "#00f5c0",
        "btn_shadow":   "rgba(0,229,176,0.3)",
        "divider":      "rgba(0,229,176,0.08)",
        "scrollbar":    "rgba(0,229,176,0.2)",
        "stat_color":   "#00e5b0",
        "sidebar_head": "#00e5b0",
        "logo_color":   "#00e5b0",
        "fonts":        "@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');",
        "font_display": "'Space Mono', monospace",
        "font_body":    "'DM Sans', sans-serif",
    },

    # ── 2. Clinical White — bright clean medical ──────────────────────────
    "Clinical White": {
        "bg":           "#f0f4f8",
        "surface":      "#e2e8f0",
        "card":         "#ffffff",
        "border":       "rgba(37,99,235,0.14)",
        "border_h":     "rgba(37,99,235,0.38)",
        "accent":       "#2563eb",
        "accent_dim":   "rgba(37,99,235,0.07)",
        "text":         "#0f172a",
        "text_muted":   "rgba(71,85,105,0.75)",
        "input_bg":     "#ffffff",
        "input_bd":     "rgba(37,99,235,0.2)",
        "input_focus":  "rgba(37,99,235,0.5)",
        "btn_fg":       "#ffffff",
        "btn_g1":       "#1d4ed8",
        "btn_g2":       "#2563eb",
        "btn_g3":       "#3b82f6",
        "btn_shadow":   "rgba(37,99,235,0.28)",
        "divider":      "rgba(37,99,235,0.1)",
        "scrollbar":    "rgba(37,99,235,0.2)",
        "stat_color":   "#2563eb",
        "sidebar_head": "#1d4ed8",
        "logo_color":   "#2563eb",
        "fonts":        "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');",
        "font_display": "'IBM Plex Sans', sans-serif",
        "font_body":    "'IBM Plex Sans', sans-serif",
    },

    # ── 3. Midnight Blue — deep navy clinical ─────────────────────────────
    "Midnight Blue": {
        "bg":           "#060d1f",
        "surface":      "#0b1530",
        "card":         "#101e42",
        "border":       "rgba(96,165,250,0.13)",
        "border_h":     "rgba(96,165,250,0.35)",
        "accent":       "#60a5fa",
        "accent_dim":   "rgba(96,165,250,0.07)",
        "text":         "#dde8ff",
        "text_muted":   "rgba(148,180,235,0.5)",
        "input_bg":     "rgba(96,165,250,0.05)",
        "input_bd":     "rgba(96,165,250,0.18)",
        "input_focus":  "rgba(96,165,250,0.45)",
        "btn_fg":       "#020c22",
        "btn_g1":       "#1d6fd8",
        "btn_g2":       "#60a5fa",
        "btn_g3":       "#93c5fd",
        "btn_shadow":   "rgba(96,165,250,0.28)",
        "divider":      "rgba(96,165,250,0.08)",
        "scrollbar":    "rgba(96,165,250,0.2)",
        "stat_color":   "#60a5fa",
        "sidebar_head": "#60a5fa",
        "logo_color":   "#60a5fa",
        "fonts":        "@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');",
        "font_display": "'Rajdhani', sans-serif",
        "font_body":    "'Inter', sans-serif",
    },

    # ── 4. Deep Purple — radiology dark violet ────────────────────────────
    "Deep Purple": {
        "bg":           "#0d0618",
        "surface":      "#130a24",
        "card":         "#1a0d32",
        "border":       "rgba(167,139,250,0.14)",
        "border_h":     "rgba(167,139,250,0.35)",
        "accent":       "#a78bfa",
        "accent_dim":   "rgba(167,139,250,0.08)",
        "text":         "#ede0ff",
        "text_muted":   "rgba(192,168,240,0.45)",
        "input_bg":     "rgba(167,139,250,0.05)",
        "input_bd":     "rgba(167,139,250,0.18)",
        "input_focus":  "rgba(167,139,250,0.45)",
        "btn_fg":       "#0d0020",
        "btn_g1":       "#7c3aed",
        "btn_g2":       "#a78bfa",
        "btn_g3":       "#c4b5fd",
        "btn_shadow":   "rgba(167,139,250,0.28)",
        "divider":      "rgba(167,139,250,0.08)",
        "scrollbar":    "rgba(167,139,250,0.2)",
        "stat_color":   "#a78bfa",
        "sidebar_head": "#a78bfa",
        "logo_color":   "#a78bfa",
        "fonts":        "@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Inter:wght@300;400;500&display=swap');",
        "font_display": "'Orbitron', sans-serif",
        "font_body":    "'Inter', sans-serif",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# THEME PICKER — call inside render_sidebar()
# ══════════════════════════════════════════════════════════════════════════════
def render_theme_picker():
    """Render theme selector in the sidebar and persist choice to session_state."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "PulmoSence"

    theme_names = list(THEMES.keys())

    st.sidebar.markdown("""
    <div style="font-family:monospace; font-size:9px; letter-spacing:3px;
                color:rgba(160,200,180,0.4); margin-bottom:8px;">
        INTERFACE THEME
    </div>
    """, unsafe_allow_html=True)

    # Colour dot previews
    dots = {
        "PulmoSence":     "#00e5b0",
        "Clinical White": "#2563eb",
        "Midnight Blue":  "#60a5fa",
        "Deep Purple":    "#a78bfa",
    }
    selected = st.sidebar.radio(
        "theme_radio",
        theme_names,
        index=theme_names.index(st.session_state["theme"]),
        label_visibility="collapsed",
        format_func=lambda name: f"● {name}",
    )
    st.session_state["theme"] = selected

    # Show accent colour dot for selected theme
    dot_color = dots.get(selected, "#ffffff")
    st.sidebar.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-top:-8px; margin-bottom:4px;
                padding:6px 10px; border-radius:8px; background:{dot_color}14;
                border:1px solid {dot_color}30;">
        <div style="width:10px; height:10px; border-radius:50%; background:{dot_color};
                    box-shadow:0 0 8px {dot_color}80; flex-shrink:0;"></div>
        <span style="font-family:monospace; font-size:9px; letter-spacing:1px;
                     color:{dot_color}99;">{selected.upper()}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# APPLY THEME — call once at the top of main()
# ══════════════════════════════════════════════════════════════════════════════
def apply_theme():
    """
    Inject theme CSS variables + full component overrides into the Streamlit app.
    Must be called after st.set_page_config() and after render_theme_picker().
    """
    name = st.session_state.get("theme", "PulmoSence")
    t    = THEMES[name]

    st.markdown(f"""
    <style>
    {t['fonts']}

    :root {{
        --bg:           {t['bg']};
        --surface:      {t['surface']};
        --card:         {t['card']};
        --border:       {t['border']};
        --border-h:     {t['border_h']};
        --accent:       {t['accent']};
        --accent-dim:   {t['accent_dim']};
        --text:         {t['text']};
        --text-muted:   {t['text_muted']};
        --input-bg:     {t['input_bg']};
        --input-bd:     {t['input_bd']};
        --input-focus:  {t['input_focus']};
        --btn-fg:       {t['btn_fg']};
        --btn-g1:       {t['btn_g1']};
        --btn-g2:       {t['btn_g2']};
        --btn-g3:       {t['btn_g3']};
        --btn-shadow:   {t['btn_shadow']};
        --divider:      {t['divider']};
        --scrollbar:    {t['scrollbar']};
        --stat-color:   {t['stat_color']};
        --sidebar-head: {t['sidebar_head']};
        --logo-color:   {t['logo_color']};
        --font-display: {t['font_display']};
        --font-body:    {t['font_body']};
    }}

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: var(--font-body) !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    #MainMenu, footer {{ visibility: hidden; }}
    [data-testid="stDecoration"] {{ display: none; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 4px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: var(--scrollbar); border-radius: 2px; }}

    /* ── Divider ── */
    hr {{ border: none !important; border-top: 1px solid var(--divider) !important; margin: 24px 0 !important; }}

    /* ── Cards ── */
    .ps-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 32px rgba(0,0,0,0.35);
        transition: border-color 0.3s;
    }}
    .ps-card:hover {{ border-color: var(--border-h); }}

    /* ── Labels ── */
    .ps-label {{
        font-family: var(--font-display);
        font-size: 9px;
        letter-spacing: 3px;
        color: var(--text-muted);
        margin-bottom: 14px;
        text-transform: uppercase;
    }}

    /* ── Stats grid ── */
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4,1fr);
        gap: 12px;
        margin-bottom: 28px;
    }}
    .stat-chip {{
        background: var(--accent-dim);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        transition: all 0.3s;
    }}
    .stat-chip:hover {{ border-color: var(--border-h); }}
    .stat-val {{
        font-family: var(--font-display);
        font-size: 22px;
        font-weight: 700;
        color: var(--stat-color);
    }}
    .stat-lbl {{ font-size: 10px; font-weight: 600; color: var(--text); margin-top: 3px; opacity: 0.65; }}
    .stat-sub {{ font-family: var(--font-display); font-size: 9px; color: var(--accent); margin-top: 2px; opacity: 0.3; }}

    /* ── Metric row ── */
    .metric-row {{
        display: grid;
        grid-template-columns: repeat(4,1fr);
        gap: 12px;
        margin: 20px 0;
    }}
    .metric-card {{
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
    }}
    .metric-val {{ font-family: var(--font-display); font-size: 22px; font-weight: 700; margin: 0; }}
    .metric-lbl {{ font-size: 9px; color: var(--text-muted); margin-top: 4px; letter-spacing: 1px; font-family: var(--font-display); }}

    /* ── Probability bars ── */
    .prob-row {{ margin-bottom: 14px; }}
    .prob-header {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; font-family: var(--font-display); }}
    .prob-track {{ height: 5px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }}
    .prob-fill  {{ height: 100%; border-radius: 3px; }}

    /* ── Nodule rows ── */
    .nodule-row {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 14px; border-radius: 10px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 8px;
        font-family: var(--font-display); font-size: 11px;
    }}

    /* ── Input overrides ── */
    .stTextInput input, .stNumberInput input {{
        background: var(--input-bg) !important;
        border: 1px solid var(--input-bd) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: var(--font-display) !important;
        font-size: 12px !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: var(--input-focus) !important;
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 15%, transparent) !important;
    }}
    .stTextInput > label, .stNumberInput > label,
    .stSelectbox > label, .stTextArea > label,
    .stDateInput > label {{
        font-family: var(--font-display) !important;
        font-size: 10px !important;
        letter-spacing: 2px !important;
        color: var(--text-muted) !important;
    }}
    textarea {{
        background: var(--input-bg) !important;
        border: 1px solid var(--input-bd) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: var(--font-display) !important;
        font-size: 11px !important;
    }}
    [data-baseweb="select"] > div {{
        background: var(--input-bg) !important;
        border: 1px solid var(--input-bd) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }}

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg,{t['btn_g1']},{t['btn_g2']},{t['btn_g3']}) !important;
        color: {t['btn_fg']} !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        letter-spacing: 3px !important;
        box-shadow: 0 0 28px {t['btn_shadow']} !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 0 46px {t['btn_shadow']} !important;
        transform: translateY(-2px) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: var(--accent-dim) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        font-family: var(--font-display) !important;
        font-size: 11px !important;
        letter-spacing: 2px !important;
        width: 100% !important;
        opacity: 0.7;
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        background: var(--accent-dim) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: 14px !important;
        padding: 16px !important;
        transition: all 0.3s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--border-h) !important;
    }}

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {{
        background: var(--accent-dim) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        font-family: var(--font-display) !important;
        font-size: 11px !important;
        letter-spacing: 2px !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }}
    [data-testid="stDownloadButton"] > button:hover {{
        border-color: var(--border-h) !important;
        box-shadow: 0 0 18px {t['btn_shadow']} !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{ background: transparent !important; gap: 4px !important; }}
    .stTabs [data-baseweb="tab"] {{
        background: var(--accent-dim) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-muted) !important;
        font-family: var(--font-display) !important;
        font-size: 10px !important;
        letter-spacing: 1px !important;
        padding: 6px 14px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: color-mix(in srgb, var(--accent) 14%, transparent) !important;
        border-color: var(--border-h) !important;
        color: var(--accent) !important;
    }}

    /* ── Slider ── */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
        background: var(--accent) !important;
    }}
    [data-testid="stSlider"] div[data-testid="stTickBar"] {{
        color: var(--text-muted) !important;
    }}

    /* ── Image captions ── */
    [data-testid="caption"] {{
        font-family: var(--font-display) !important;
        font-size: 9px !important;
        letter-spacing: 1px !important;
        color: var(--text-muted) !important;
        text-align: center !important;
    }}

    /* ── Alert / warning ── */
    .stAlert {{
        background: rgba(240,180,41,0.07) !important;
        border: 1px solid rgba(240,180,41,0.22) !important;
        border-radius: 10px !important;
    }}

    /* ── Sidebar headings ── */
    [data-testid="stSidebar"] h3 {{
        font-family: var(--font-display) !important;
        font-size: 10px !important;
        letter-spacing: 2px !important;
        color: var(--sidebar-head) !important;
        text-transform: uppercase !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {{
        font-size: 12px !important;
        color: var(--text-muted) !important;
        font-family: var(--font-body) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

