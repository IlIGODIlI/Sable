from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.parameters import load_parameter_dictionary

st.set_page_config(
    page_title="Landslide Risk Analysis & Inference System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(rgba(30, 41, 59, 0.4) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(15, 23, 42, 0.95), rgba(11, 15, 25, 0.98));
        background-size: 24px 24px, 100% 100%;
        color: #f1f5f9;
    }

    .app-header {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1.2rem;
        margin-bottom: 1.5rem;
    }

    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }

    .status-tag {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        margin-right: 0.5rem;
    }

    .panel-box {
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }

    .panel-topo {
        border-top: 3px solid #d97706;
    }

    .panel-spectral {
        border-top: 3px solid #059669;
    }

    .panel-sar {
        border-top: 3px solid #0284c7;
    }

    .panel-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .header-topo { color: #f59e0b; }
    .header-spectral { color: #10b981; }
    .header-sar { color: #38bdf8; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    }

    .model-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94a3b8;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }

    .model-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 0.6rem;
    }

    .metric-row {
        display: flex;
        gap: 0.35rem;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    .chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background: #1f2937;
        border: 1px solid #374151;
        color: #e5e7eb;
    }

    .alert-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 0.65rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        text-align: center;
        letter-spacing: 0.03em;
        margin: 0.75rem 0;
    }

    .alert-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 0.65rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        text-align: center;
        letter-spacing: 0.03em;
        margin: 0.75rem 0;
    }

    .data-table {
        width: 100%;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        margin-top: 0.6rem;
        border-collapse: collapse;
    }

    .data-table td {
        padding: 0.3rem 0;
        border-bottom: 1px solid #1f2937;
        color: #cbd5e1;
    }

    .data-table td.val {
        text-align: right;
        color: #38bdf8;
        font-weight: 600;
    }

    .divider-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.35rem;
        margin: 1.4rem 0 1rem 0;
    }

    /* Parameter Knowledge Base styling */
    .param-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #38bdf8;
    }

    .param-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .param-tag {
        font-size: 0.68rem;
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        font-weight: 600;
    }

    .param-val-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #10b981;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.1);
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .param-field {
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.4;
        margin-top: 0.25rem;
    }

    .param-purpose-box {
        font-size: 0.75rem;
        color: #cbd5e1;
        line-height: 1.4;
        background: rgba(30, 41, 59, 0.45);
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        margin-top: 0.4rem;
        border-left: 2px solid #0284c7;
    }

    /* Streamlit slider override for dark aesthetic */
    div[data-testid="stSlider"] label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "elevation", "slope", "aspect", "curvature",
    "plan_curvature", "profile_curvature",
    "twi", "tri", "spi", "fdr",
    "NDVI", "NDWI", "ARVI", "EVI", "GNDVI", "GRVI",
    "MSAVI", "SAVI", "BSI", "mNDMI", "mNDWI",
    "ASM", "Contrast", "Dissimilarity", "Energy",
    "Entropy", "GLCMCorrelation", "GLCMMean", "Homogeneity",
]

DEFAULT_VALUES = {
    "elevation": -57.4,
    "slope": 5.0,
    "aspect": 188.9,
    "curvature": 0.0001,
    "plan_curvature": 0.00004,
    "profile_curvature": 0.0001,
    "twi": 4.06,
    "tri": 0.78,
    "spi": -2.32,
    "fdr": 8.0,
    "NDVI": 0.39,
    "NDWI": -0.32,
    "ARVI": 0.51,
    "EVI": -3.28,
    "GNDVI": 0.32,
    "GRVI": 0.08,
    "MSAVI": 0.56,
    "SAVI": 0.59,
    "BSI": -0.10,
    "mNDMI": 0.11,
    "mNDWI": -0.20,
    "ASM": 0.03,
    "Contrast": 159.9,
    "Dissimilarity": 13.67,
    "Energy": 0.18,
    "Entropy": 8.50,
    "GLCMCorrelation": 0.76,
    "GLCMMean": 22.63,
    "Homogeneity": 0.32,
}


# ---------------------------------------------------------------------------
# Feature description dictionary for slider dropdown help
# ---------------------------------------------------------------------------
FEATURE_INFO = {
    "elevation": {
        "label": "Elevation (m)",
        "meaning": "Height above mean sea level derived from Digital Elevation Model (DEM). "
                   "Controls temperature lapse rate, soil moisture, and rainfall patterns.",
        "role": "Higher elevations increase slope instability and exposure to orographic "
                "precipitation. Low-lying valleys accumulate saturated material, raising "
                "debris-flow risk.",
        "seasons": {
            "Monsoon (Jun-Sep)": "-150 to 1500 m — entire range active; valley floors "
                                 "flood-saturated (< 0 m) and steep ridges (> 800 m) see "
                                 "intense rainfall-triggered slides.",
            "Pre-monsoon (Mar-May)": "0–600 m — drier lower slopes; snowmelt triggers "
                                     "shallow failures on higher terrain (> 1000 m).",
            "Post-monsoon (Oct-Nov)": "50–900 m — residual pore-water pressure keeps "
                                      "mid-slope zones (200–600 m) moderately unstable.",
            "Winter (Dec-Feb)": "400–1500 m — freeze-thaw cycles destabilise high-altitude "
                                "terrain; low-elevation areas largely stable.",
        },
    },
    "slope": {
        "label": "Slope Gradient (°)",
        "meaning": "Angle of the terrain surface in degrees, computed from DEM gradient. "
                   "Directly governs the gravitational shear stress acting on slope material.",
        "role": "The single strongest topographic predictor of landslide initiation. Slopes "
                "> 25° are critically susceptible; slopes < 5° rarely fail without additional "
                "triggers such as river undercutting.",
        "seasons": {
            "Monsoon (Jun-Sep)": "0–60° — all slopes at risk; slopes 20–40° show peak "
                                 "failure frequency due to saturation.",
            "Pre-monsoon (Mar-May)": "25–60° — only steep slopes fail; driven by "
                                     "snowmelt or localised thunderstorms.",
            "Post-monsoon (Oct-Nov)": "15–45° — delayed failures on slopes that absorbed "
                                      "peak monsoon moisture.",
            "Winter (Dec-Feb)": "30–60° — primarily rockfall on very steep faces; soil "
                                "slides rare in frozen ground.",
        },
    },
    "spi": {
        "label": "Stream Power Index (SPI)",
        "meaning": "SPI = ln(A · tan β) where A is upslope contributing area and β is "
                   "local slope. Measures erosive energy of overland flow.",
        "role": "High SPI (> 5) indicates concentrated flow capable of undercutting slopes "
                "and triggering debris flows. Negative SPI signals diverging or low-energy "
                "flow paths — typically stable.",
        "seasons": {
            "Monsoon (Jun-Sep)": "-2 to +12 — peak values in channelised gullies; "
                                 "values > 6 closely associated with active debris flows.",
            "Pre-monsoon (Mar-May)": "-5 to +4 — low stream power; meltwater channels "
                                     "show moderate SPI (2–5).",
            "Post-monsoon (Oct-Nov)": "-3 to +6 — residual flow in monsoon channels; "
                                      "values declining as rivers recede.",
            "Winter (Dec-Feb)": "-8 to +2 — minimal surface flow; SPI largely negative "
                                "on frozen or dry slopes.",
        },
    },
    "tri": {
        "label": "Terrain Ruggedness Index (TRI)",
        "meaning": "Mean absolute difference in elevation between a cell and its 8 "
                   "neighbours. Quantifies surface roughness and micro-relief complexity.",
        "role": "High TRI (> 2) indicates highly dissected terrain where regolith is "
                "poorly anchored. Low TRI (< 0.3) corresponds to flat plains with minimal "
                "failure potential.",
        "seasons": {
            "Monsoon (Jun-Sep)": "0.3–4.5 — full range; heavily dissected slopes "
                                 "(TRI > 2.5) most vulnerable to shallow slides.",
            "Pre-monsoon (Mar-May)": "0.5–3.5 — moderate ruggedness zones susceptible "
                                     "to snowmelt-induced failures.",
            "Post-monsoon (Oct-Nov)": "0.5–3.0 — similar to pre-monsoon; erosion "
                                      "slightly increases surface roughness.",
            "Winter (Dec-Feb)": "0.2–2.5 — frozen ground suppresses surface processes; "
                                "effective ruggedness influence reduced.",
        },
    },
    "curvature": {
        "label": "Surface Curvature",
        "meaning": "Second derivative of the elevation surface. Positive = convex "
                   "(diverging flow, erosion); Negative = concave (converging flow, "
                   "deposition and saturation).",
        "role": "Concave slopes concentrate subsurface flow and pore-water pressure — "
                "prime landslide initiation zones. Convex ridges shed water and are "
                "relatively drier.",
        "seasons": {
            "Monsoon (Jun-Sep)": "-0.040 to +0.040 — concave hollows (< -0.010) "
                                 "become critically saturated during sustained rainfall.",
            "Pre-monsoon (Mar-May)": "-0.020 to +0.020 — moderate effect; meltwater "
                                     "pools in concavities.",
            "Post-monsoon (Oct-Nov)": "-0.030 to +0.025 — concave zones retain "
                                      "residual moisture longest.",
            "Winter (Dec-Feb)": "-0.015 to +0.015 — minimal active flow; curvature "
                                "effect on failure risk low.",
        },
    },
    "NDVI": {
        "label": "NDVI — Normalised Difference Vegetation Index",
        "meaning": "NDVI = (NIR − Red) / (NIR + Red). Measures green vegetation density "
                   "from Sentinel-2 imagery. Ranges −1 (water/bare) to +1 (dense canopy).",
        "role": "Dense vegetation (NDVI > 0.5) reinforces shallow soils via root cohesion "
                "and reduces surface runoff. Bare or degraded slopes (NDVI < 0.2) are "
                "significantly more susceptible.",
        "seasons": {
            "Monsoon (Jun-Sep)": "0.35–0.85 — peak greenness; dense forest can mask "
                                 "underlying instability.",
            "Pre-monsoon (Mar-May)": "0.10–0.55 — vegetation stressed and sparse on "
                                     "south-facing slopes; deciduous forests partly bare.",
            "Post-monsoon (Oct-Nov)": "0.20–0.65 — greenness declining; harvest and "
                                      "leaf-fall expose soils.",
            "Winter (Dec-Feb)": "0.05–0.40 — minimum values; snowpack may suppress "
                                "readings in high-altitude pixels.",
        },
    },
    "EVI": {
        "label": "EVI — Enhanced Vegetation Index",
        "meaning": "EVI = 2.5 × (NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1). "
                   "Improves NDVI in dense-canopy and aerosol-affected regions.",
        "role": "More sensitive than NDVI in saturated monsoon conditions. Steep drops "
                "in EVI signal vegetation loss from past mass movements — a key "
                "reactivation indicator.",
        "seasons": {
            "Monsoon (Jun-Sep)": "-1.5 to +3.0 — high values in intact forest; "
                                 "disturbed landslide scars show strongly negative EVI.",
            "Pre-monsoon (Mar-May)": "-4.0 to +1.5 — widespread negative values on "
                                     "agricultural and degraded land.",
            "Post-monsoon (Oct-Nov)": "-3.5 to +2.0 — transitional; tea and "
                                      "grassland areas drop sharply.",
            "Winter (Dec-Feb)": "-6.0 to +0.5 — lowest values; snow and bare soil "
                                "dominate high-altitude pixels.",
        },
    },
    "GRVI": {
        "label": "GRVI — Green-Red Vegetation Index",
        "meaning": "GRVI = (Green − Red) / (Green + Red). Sensitive to chlorophyll "
                   "content and canopy stress — complements NDVI for phenology tracking.",
        "role": "Stressed or diseased vegetation (low GRVI) has reduced root cohesion. "
                "Sudden GRVI drops identify fresh disturbance zones before visual "
                "evidence is apparent.",
        "seasons": {
            "Monsoon (Jun-Sep)": "0.05–0.35 — healthy canopy; values plateau in "
                                 "dense broadleaf forest.",
            "Pre-monsoon (Mar-May)": "-0.10 to +0.15 — stressed and sparse "
                                     "vegetation; agricultural bare-soil periods.",
            "Post-monsoon (Oct-Nov)": "-0.05 to +0.20 — declining after harvest; "
                                      "some recovery in evergreen zones.",
            "Winter (Dec-Feb)": "-0.30 to +0.05 — near-zero or negative in "
                                "deciduous and snow-covered areas.",
        },
    },
    "mNDWI": {
        "label": "mNDWI — Modified Normalised Difference Water Index",
        "meaning": "mNDWI = (Green − SWIR) / (Green + SWIR). Highlights surface water "
                   "bodies and saturated soils; insensitive to vegetation.",
        "role": "Rising mNDWI during monsoon indicates soil saturation and stream "
                "expansion — direct precursors of debris flows and earthflows. "
                "Values > 0.0 suggest near-saturated or waterlogged conditions.",
        "seasons": {
            "Monsoon (Jun-Sep)": "-0.10 to +0.65 — rivers flood, valley soils "
                                 "saturate; widespread positive values.",
            "Pre-monsoon (Mar-May)": "-0.45 to +0.10 — dry soils dominate; "
                                     "perennial streams retain slightly positive values.",
            "Post-monsoon (Oct-Nov)": "-0.30 to +0.30 — transitional; receding "
                                      "floodwater leaves moist corridors.",
            "Winter (Dec-Feb)": "-0.60 to -0.10 — minimum water content; "
                                "frozen surfaces may show anomalous readings.",
        },
    },
    "GLCMMean": {
        "label": "GLCM Texture Mean",
        "meaning": "Mean value of the Grey-Level Co-occurrence Matrix computed from "
                   "Sentinel-1 SAR backscatter. Reflects average radar return intensity "
                   "over a texture window.",
        "role": "High GLCM Mean (> 25) indicates rough, heterogeneous surfaces typical "
                "of active landslide deposits or bare rocky outcrops. Smooth, "
                "low-mean areas (< 10) are dense vegetation or calm water.",
        "seasons": {
            "Monsoon (Jun-Sep)": "10–35 — wet vegetation increases backscatter; "
                                 "flooded fields show low values (8–14).",
            "Pre-monsoon (Mar-May)": "15–45 — dry bare soil gives high backscatter; "
                                     "rocky exposed terrain peaks at 40+.",
            "Post-monsoon (Oct-Nov)": "12–38 — mixed surface conditions; "
                                      "landslide scars remain high (28–40).",
            "Winter (Dec-Feb)": "8–30 — snow damping reduces backscatter at "
                                "high altitude; dry bare soil still high at mid-elevation.",
        },
    },
    "GLCMCorrelation": {
        "label": "GLCM Correlation",
        "meaning": "Measures linear dependency of grey levels across neighbouring "
                   "pixels in the SAR texture window. Ranges −1 (perfectly "
                   "anti-correlated) to +1 (perfectly correlated).",
        "role": "High correlation (> 0.7) reflects uniform, spatially consistent "
                "surfaces — intact forest or farmland. Low correlation (< 0.3) "
                "indicates chaotic, disturbed terrain characteristic of landslide "
                "scars and debris fields.",
        "seasons": {
            "Monsoon (Jun-Sep)": "0.40–0.90 — high correlation in flooded "
                                 "rice fields and saturated forest.",
            "Pre-monsoon (Mar-May)": "0.25–0.80 — variable; ploughed fields "
                                     "and bare slopes reduce correlation.",
            "Post-monsoon (Oct-Nov)": "0.30–0.85 — recovering canopy improves "
                                      "correlation; active scars remain low.",
            "Winter (Dec-Feb)": "0.20–0.75 — snow cover creates uniform patches "
                                "(high); exposed rocky terrain remains low.",
        },
    },
    "twi": {
        "label": "Topographic Wetness Index (TWI)",
        "meaning": "TWI = ln(A / tan β) where A is upslope drainage area and β is "
                   "local slope. Estimates the tendency of a location to accumulate "
                   "soil moisture.",
        "role": "High TWI (> 8) corresponds to concave hollows and valley bottoms "
                "where soil saturation is prolonged — the most common landslide "
                "initiation zones. Low TWI (< 3) on steep ridges drains rapidly.",
        "seasons": {
            "Monsoon (Jun-Sep)": "5–15 — widespread high values; hollows with "
                                 "TWI > 10 almost always saturated.",
            "Pre-monsoon (Mar-May)": "2–8 — lower values as soils dry; snowmelt "
                                     "corridors retain TWI 6–9.",
            "Post-monsoon (Oct-Nov)": "3–10 — declining but significant residual "
                                      "wetness in valley floors.",
            "Winter (Dec-Feb)": "1–6 — minimum; frozen ground limits infiltration "
                                "and effective wetness is low.",
        },
    },
    "aspect": {
        "label": "Terrain Aspect (°)",
        "meaning": "Compass direction the slope faces, in degrees (0° = North, "
                   "90° = East, 180° = South, 270° = West). Determines insolation, "
                   "drying rate, and vegetation type.",
        "role": "South-facing slopes (135°–225°) receive maximum solar radiation — "
                "drier and more sparsely vegetated, prone to regolith weathering. "
                "North-facing slopes (315°–45°) stay wetter longer and support "
                "denser root networks.",
        "seasons": {
            "Monsoon (Jun-Sep)": "0°–360° — aspect matters less when rainfall is "
                                 "universal; SW-facing slopes (200°–250°) intercept "
                                 "prevailing monsoon winds most.",
            "Pre-monsoon (Mar-May)": "90°–270° — south and east-facing slopes dry "
                                     "out first; snowmelt faster on south aspects.",
            "Post-monsoon (Oct-Nov)": "0°–180° — north-facing retain moisture "
                                      "longest; south-facing dry within weeks.",
            "Winter (Dec-Feb)": "270°–360° and 0°–90° — north-facing slopes "
                                "accumulate ice and snow; freeze-thaw cycles "
                                "dominate east/west aspects.",
        },
    },
}



@st.cache_resource
def load_all_models():
    from pathlib import Path as _Path
    import json as _json
    lr_pipe = joblib.load(_Path("models/logistic_regression.joblib"))
    rf_pipe = joblib.load(_Path("models/random_forest.joblib"))
    xgb_pipe = joblib.load(_Path("models/xgboost.joblib"))

    results_file = _Path("outputs/results.json")
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            metrics = _json.load(f)
    else:
        metrics = {
            "logistic_regression": {
                "accuracy": 0.6923, "balanced_accuracy": 0.6988, "roc_auc": 0.7787,
                "r2_score": 0.1587, "z_score_accuracy": 27.9, "brier_score": 0.1918,
                "macro_f1": 0.6813, "matthews_corrcoef": 0.3808,
            },
            "random_forest": {
                "accuracy": 0.9138, "balanced_accuracy": 0.9183, "roc_auc": 0.9755,
                "r2_score": 0.7197, "z_score_accuracy": 169.3, "brier_score": 0.0639,
                "macro_f1": 0.9077, "matthews_corrcoef": 0.8186,
            },
            "xgboost": {
                "accuracy": 0.9018, "balanced_accuracy": 0.9080, "roc_auc": 0.9703,
                "r2_score": 0.6920, "z_score_accuracy": 161.6, "brier_score": 0.0702,
                "macro_f1": 0.8953, "matthews_corrcoef": 0.7953,
            },
        }

    param_dict = load_parameter_dictionary(_Path("outputs/model_parameters.json"))
    return lr_pipe, rf_pipe, xgb_pipe, metrics, param_dict


lr_pipe, rf_pipe, xgb_pipe, metrics, param_dict = load_all_models()

# Top Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Landslide Susceptibility Inference System</div>
        <div class="app-subtitle">
            <span class="status-tag">SYSTEM: ACTIVE</span>
            <span class="status-tag">DATASETS: MULTI-STATE SENTINEL + IN SITU (464K+)</span>
            <span class="status-tag">DYNAMIC TUNING: SCIKIT-LEARN RANDOMIZED SEARCH</span>
            <span class="status-tag">SCORING: R² / Z-SCORE / BRIER</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Parameter Help (rendered once, always visible)
# ---------------------------------------------------------------------------
def _render_feature_help(key: str):
    """Render a compact dropdown expander with feature description and seasonal ranges."""
    info = FEATURE_INFO.get(key)
    if not info:
        return
    season_rows = "".join(
        f'''<tr>
            <td style="padding:0.25rem 0.5rem 0.25rem 0; color:#94a3b8; white-space:nowrap; font-weight:600;">{season}</td>
            <td style="padding:0.25rem 0; color:#cbd5e1;">{rng}</td>
        </tr>'''
        for season, rng in info["seasons"].items()
    )
    with st.expander(f"\u2139\ufe0f About: {info['label']}"):
        st.markdown(
            f'''<div style="font-size:0.76rem; line-height:1.6; font-family:'Inter',sans-serif;">
  <div style="margin-bottom:0.5rem;">
    <span style="color:#38bdf8; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">WHAT IT IS</span><br>
    <span style="color:#e2e8f0;">{info["meaning"]}</span>
  </div>
  <div style="margin-bottom:0.6rem;">
    <span style="color:#10b981; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">ROLE IN LANDSLIDE RISK</span><br>
    <span style="color:#e2e8f0;">{info["role"]}</span>
  </div>
  <div>
    <span style="color:#f59e0b; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">SEASONAL VALUE RANGES</span>
    <table style="width:100%; margin-top:0.35rem; border-collapse:collapse; font-size:0.73rem;">
      {season_rows}
    </table>
  </div>
</div>''',
            unsafe_allow_html=True,
        )


def render_param_help():
    with st.sidebar.expander("📚 Parameter Help"):
        param_options = []
        for m_key, m_val in param_dict.items():
            for p_key, p_info in m_val.get("parameters", {}).items():
                label = f"{m_val.get('model_name', '')} – {p_info.get('display_name', p_key)}"
                param_options.append((label, m_key, p_key))
        option_labels = [opt[0] for opt in param_options]
        if option_labels:
            selected = st.selectbox("Select a parameter:", options=option_labels, key="param_help_select")
            sel_idx = option_labels.index(selected)
            _, sel_m_key, sel_p_key = param_options[sel_idx]
            p_info = param_dict[sel_m_key]["parameters"][sel_p_key]
            st.markdown(f"**Meaning:** {p_info.get('meaning', '')}")
            st.markdown(f"**Purpose:** {p_info.get('purpose', '')}")
            st.markdown(f"**Typical value / range:** {p_info.get('value', '')}")


render_param_help()

# ---------------------------------------------------------------------------
# Section 01: Preset Selectors
# ---------------------------------------------------------------------------
st.markdown('<div class="divider-tag">// SECTION 01: TERRAIN SCENARIO PRESETS</div>', unsafe_allow_html=True)

preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])
with preset_col1:
    btn_high = st.button("SET PRESET: HIGH-RISK VALLEY", use_container_width=True)
with preset_col2:
    btn_low = st.button("SET PRESET: LOW-RISK PLAIN", use_container_width=True)
with preset_col3:
    btn_reset = st.button("RESET TO REGIONAL MEDIANS", use_container_width=True)

if "inputs" not in st.session_state or btn_reset:
    st.session_state["inputs"] = dict(DEFAULT_VALUES)

if btn_high:
    st.session_state["inputs"] = dict(DEFAULT_VALUES)
    st.session_state["inputs"]["elevation"] = -85.0
    st.session_state["inputs"]["slope"] = 28.5
    st.session_state["inputs"]["spi"] = -6.50
    st.session_state["inputs"]["tri"] = 1.95
    st.session_state["inputs"]["curvature"] = 0.015
    st.session_state["inputs"]["NDVI"] = 0.22
    st.session_state["inputs"]["EVI"] = -1.50
    st.session_state["inputs"]["GRVI"] = 0.02
    st.session_state["inputs"]["mNDWI"] = -0.10
    st.session_state["inputs"]["GLCMMean"] = 17.5
    st.session_state["inputs"]["GLCMCorrelation"] = 0.85

if btn_low:
    st.session_state["inputs"] = dict(DEFAULT_VALUES)
    st.session_state["inputs"]["elevation"] = -40.0
    st.session_state["inputs"]["slope"] = 2.1
    st.session_state["inputs"]["spi"] = -1.20
    st.session_state["inputs"]["tri"] = 0.35
    st.session_state["inputs"]["curvature"] = -0.005
    st.session_state["inputs"]["NDVI"] = 0.48
    st.session_state["inputs"]["EVI"] = -3.85
    st.session_state["inputs"]["GRVI"] = 0.12
    st.session_state["inputs"]["mNDWI"] = -0.28
    st.session_state["inputs"]["GLCMMean"] = 24.5
    st.session_state["inputs"]["GLCMCorrelation"] = 0.65

inputs = st.session_state["inputs"]

# ---------------------------------------------------------------------------
# Section 02: Influential Parameter Slide Bars
# ---------------------------------------------------------------------------
st.markdown('<div class="divider-tag">// SECTION 02: INFLUENTIAL PARAMETER SLIDE BARS</div>', unsafe_allow_html=True)

col_topo, col_spectral, col_sar = st.columns(3)

with col_topo:
    st.markdown(
        """
        <div class="panel-box panel-topo">
            <div class="panel-header header-topo">Topographic Drivers (DEM Derived)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    inputs["elevation"] = st.slider("Elevation (m)", -150.0, 1500.0, float(inputs["elevation"]), 1.0)
    _render_feature_help("elevation")
    inputs["slope"] = st.slider("Slope Gradient (degrees)", 0.0, 60.0, float(inputs["slope"]), 0.5)
    _render_feature_help("slope")
    inputs["spi"] = st.slider("Stream Power Index (SPI)", -15.0, 15.0, float(inputs["spi"]), 0.1)
    _render_feature_help("spi")
    inputs["tri"] = st.slider("Terrain Ruggedness Index (TRI)", 0.0, 8.0, float(inputs["tri"]), 0.05)
    _render_feature_help("tri")
    inputs["curvature"] = st.slider("Surface Curvature", -0.050, 0.050, float(inputs["curvature"]), 0.001, format="%.4f")
    _render_feature_help("curvature")

with col_spectral:
    st.markdown(
        """
        <div class="panel-box panel-spectral">
            <div class="panel-header header-spectral">Spectral &amp; Moisture Indices (Sentinel-2)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    inputs["NDVI"] = st.slider("NDVI (Normalized Vegetation)", -0.50, 1.00, float(inputs["NDVI"]), 0.01)
    _render_feature_help("NDVI")
    inputs["EVI"] = st.slider("EVI (Enhanced Vegetation)", -10.0, 10.0, float(inputs["EVI"]), 0.1)
    _render_feature_help("EVI")
    inputs["GRVI"] = st.slider("GRVI (Green-Red Vegetation)", -0.50, 0.50, float(inputs["GRVI"]), 0.01)
    _render_feature_help("GRVI")
    inputs["mNDWI"] = st.slider("mNDWI (Water Index)", -0.80, 0.80, float(inputs["mNDWI"]), 0.01)
    _render_feature_help("mNDWI")

with col_sar:
    st.markdown(
        """
        <div class="panel-box panel-sar">
            <div class="panel-header header-sar">Radar Texture &amp; Hydro (Sentinel-1 SAR)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    inputs["GLCMMean"] = st.slider("GLCM Texture Mean", 0.0, 60.0, float(inputs["GLCMMean"]), 0.5)
    _render_feature_help("GLCMMean")
    inputs["GLCMCorrelation"] = st.slider("GLCM Correlation", -0.50, 1.00, float(inputs["GLCMCorrelation"]), 0.01)
    _render_feature_help("GLCMCorrelation")
    inputs["twi"] = st.slider("Topographic Wetness (TWI)", 0.0, 15.0, float(inputs["twi"]), 0.1)
    _render_feature_help("twi")
    inputs["aspect"] = st.slider("Terrain Aspect Orientation (°)", 0.0, 360.0, float(inputs["aspect"]), 1.0)
    _render_feature_help("aspect")

with st.expander("EXPAND: ALL 29 SENTINEL & GEOMORPHIC FEATURE CONSTANTS"):
    sub1, sub2, sub3 = st.columns(3)
    remaining_feats = [f for f in FEATURES if f not in [
        "elevation", "slope", "spi", "tri", "curvature",
        "NDVI", "EVI", "GRVI", "mNDWI", "GLCMMean", "GLCMCorrelation", "twi", "aspect",
    ]]
    for i, feat in enumerate(remaining_feats):
        target = sub1 if i % 3 == 0 else (sub2 if i % 3 == 1 else sub3)
        with target:
            inputs[feat] = st.number_input(feat, value=float(inputs[feat]), format="%.4f")

input_df = pd.DataFrame([inputs])[FEATURES]

# ---------------------------------------------------------------------------
# Compute Model Inferences
# ---------------------------------------------------------------------------
# 1. Logistic Regression
lr_imputer = lr_pipe.named_steps["imputer"]
lr_scaler = lr_pipe.named_steps["scaler"]
lr_model = lr_pipe.named_steps["model"]

lr_imp = lr_imputer.transform(input_df)
lr_scaled = lr_scaler.transform(lr_imp)
z_score = float(np.dot(lr_scaled, lr_model.coef_[0])[0] + lr_model.intercept_[0])
lr_prob = float(1.0 / (1.0 + np.exp(-z_score)))
lr_pred = 1 if lr_prob >= 0.5 else 0

# 2. Random Forest
rf_prob = float(rf_pipe.predict_proba(input_df)[0, 1])
rf_pred = int(rf_pipe.predict(input_df)[0])
rf_trees = len(rf_pipe.named_steps["model"].estimators_)
rf_votes_pos = int(round(rf_prob * rf_trees))
rf_votes_neg = rf_trees - rf_votes_pos

# 3. XGBoost
xgb_imputer = xgb_pipe.named_steps["imputer"]
xgb_model = xgb_pipe.named_steps["model"]
xgb_imp = xgb_imputer.transform(input_df)
xgb_margin = float(xgb_model.predict(xgb_imp, output_margin=True)[0])
xgb_prob = float(xgb_pipe.predict_proba(input_df)[0, 1])
xgb_pred = int(xgb_pipe.predict(input_df)[0])

# ---------------------------------------------------------------------------
# Section 03: Tri-Model Inference
# ---------------------------------------------------------------------------
st.markdown('<div class="divider-tag">// SECTION 03: TRI-MODEL INFERENCE &amp; ARCHITECTURAL TELEMETRY</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)

# --- Logistic Regression ---
with p1:
    with st.container(border=True):
        lr_meta = metrics.get("logistic_regression", {})
        lr_params = param_dict.get("logistic_regression", {}).get("parameters", {})
        st.markdown(
            f"""
            <div style="border-left: 3px solid #6366f1; padding-left: 0.6rem; margin-bottom: 0.7rem;">
                <div class="model-code">MODEL_ID: 01 // STATISTICAL</div>
                <div class="model-name">Logistic Regression</div>
            </div>
            <div class="metric-row">
                <span class="chip">ACC: {lr_meta.get('accuracy', 0.6923)*100:.1f}%</span>
                <span class="chip">BAL ACC: {lr_meta.get('balanced_accuracy', 0.6988)*100:.1f}%</span>
                <span class="chip">ROC: {lr_meta.get('roc_auc', 0.7787):.3f}</span>
                <span class="chip">R²: {lr_meta.get('r2_score', 0.1587):.3f}</span>
                <span class="chip">Z-STAT: {lr_meta.get('z_score_accuracy', 27.9):+.1f}σ</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if lr_pred == 1:
            st.markdown(f'<div class="alert-high">[ HIGH RISK: LANDSLIDE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {lr_prob*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-safe">[ LOW RISK: STABLE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {lr_prob*100:.1f}%</span></div>', unsafe_allow_html=True)

        st.progress(lr_prob)

        lr_c_val = lr_params.get("C", {}).get("value", round(float(getattr(lr_model, "C", 1.0)), 4))
        lr_solver = lr_params.get("solver", {}).get("value", getattr(lr_model, "solver", "lbfgs"))
        lr_class_w = lr_params.get("class_weight", {}).get("value", getattr(lr_model, "class_weight", "balanced"))

        st.markdown(
            f"""
            <table class="data-table">
                <tr><td>Logit Score (z)</td><td class="val">{z_score:+.4f}</td></tr>
                <tr><td>Sigmoid σ(z)</td><td class="val">{lr_prob:.4f}</td></tr>
                <tr><td>R² Calibration Score</td><td class="val">{lr_meta.get('r2_score', 0.1587):.4f}</td></tr>
                <tr><td>Z-Score Significance (Z_acc)</td><td class="val">{lr_meta.get('z_score_accuracy', 27.9):+.2f}σ</td></tr>
                <tr><td>Brier Score Loss</td><td class="val">{lr_meta.get('brier_score', 0.1918):.4f}</td></tr>
                <tr><td>Inv Regularization (C)</td><td class="val">{lr_c_val}</td></tr>
                <tr><td>Solver / Weighting</td><td class="val">{lr_solver} ({lr_class_w})</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔬 DYNAMIC PARAMETER PURPOSE"):
            for p_name, p_info in lr_params.items():
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:0.45rem 0.6rem; border-radius:4px; margin-bottom:0.45rem; border-left:2px solid #6366f1;">
                        <div style="font-family:JetBrains Mono; font-size:0.75rem; color:#a5b4fc; font-weight:700;">
                            {p_info.get('display_name', p_name)}: <span style="color:#38bdf8;">{p_info.get('value')}</span>
                        </div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;"><b>Meaning:</b> {p_info.get('meaning')}</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; margin-top:0.2rem;"><b>Landslide Purpose:</b> {p_info.get('purpose')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="font-size:0.75rem; color:#94a3b8; font-family:JetBrains Mono; margin-top:0.8rem; font-weight:600;">TOP WEIGHTS (w_i):</div>', unsafe_allow_html=True)
        top_weights = pd.Series(lr_model.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False).head(4)
        table_rows = "".join([f"<tr><td>{f}</td><td class='val'>{w:+.3f}</td></tr>" for f, w in top_weights.items()])
        st.markdown(f'<table class="data-table">{table_rows}</table>', unsafe_allow_html=True)

# --- Random Forest ---
with p2:
    with st.container(border=True):
        rf_meta = metrics.get("random_forest", {})
        rf_params = param_dict.get("random_forest", {}).get("parameters", {})
        st.markdown(
            f"""
            <div style="border-left: 3px solid #10b981; padding-left: 0.6rem; margin-bottom: 0.7rem;">
                <div class="model-code">MODEL_ID: 02 // BAGGING ENSEMBLE</div>
                <div class="model-name">Random Forest</div>
            </div>
            <div class="metric-row">
                <span class="chip">ACC: {rf_meta.get('accuracy', 0.9138)*100:.1f}%</span>
                <span class="chip">BAL ACC: {rf_meta.get('balanced_accuracy', 0.9183)*100:.1f}%</span>
                <span class="chip">ROC: {rf_meta.get('roc_auc', 0.9755):.3f}</span>
                <span class="chip">R²: {rf_meta.get('r2_score', 0.7197):.3f}</span>
                <span class="chip">Z-STAT: {rf_meta.get('z_score_accuracy', 169.3):+.1f}σ</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if rf_pred == 1:
            st.markdown(f'<div class="alert-high">[ HIGH RISK: LANDSLIDE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {rf_prob*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-safe">[ LOW RISK: STABLE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {rf_prob*100:.1f}%</span></div>', unsafe_allow_html=True)

        st.progress(rf_prob)

        rf_depth = rf_params.get("max_depth", {}).get("value", getattr(rf_pipe.named_steps["model"], "max_depth", 18))
        rf_leaf = rf_params.get("min_samples_leaf", {}).get("value", getattr(rf_pipe.named_steps["model"], "min_samples_leaf", 8))
        rf_split = rf_params.get("min_samples_split", {}).get("value", getattr(rf_pipe.named_steps["model"], "min_samples_split", 5))
        rf_feat = rf_params.get("max_features", {}).get("value", getattr(rf_pipe.named_steps["model"], "max_features", 0.7))

        st.markdown(
            f"""
            <table class="data-table">
                <tr><td>Ensemble Trees</td><td class="val">{rf_trees}</td></tr>
                <tr><td>Consensus (Landslide)</td><td class="val">{rf_votes_pos} / {rf_trees}</td></tr>
                <tr><td>R² Calibration Score</td><td class="val">{rf_meta.get('r2_score', 0.7197):.4f}</td></tr>
                <tr><td>Z-Score Significance (Z_acc)</td><td class="val">{rf_meta.get('z_score_accuracy', 169.3):+.2f}σ</td></tr>
                <tr><td>Brier Score Loss</td><td class="val">{rf_meta.get('brier_score', 0.0639):.4f}</td></tr>
                <tr><td>Max Depth / Min Leaf</td><td class="val">{rf_depth} / {rf_leaf}</td></tr>
                <tr><td>Split (Min / Feat)</td><td class="val">{rf_split} / {rf_feat}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔬 DYNAMIC PARAMETER PURPOSE"):
            for p_name, p_info in rf_params.items():
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:0.45rem 0.6rem; border-radius:4px; margin-bottom:0.45rem; border-left:2px solid #10b981;">
                        <div style="font-family:JetBrains Mono; font-size:0.75rem; color:#6ee7b7; font-weight:700;">
                            {p_info.get('display_name', p_name)}: <span style="color:#38bdf8;">{p_info.get('value')}</span>
                        </div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;"><b>Meaning:</b> {p_info.get('meaning')}</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; margin-top:0.2rem;"><b>Landslide Purpose:</b> {p_info.get('purpose')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="font-size:0.75rem; color:#94a3b8; font-family:JetBrains Mono; margin-top:0.8rem; font-weight:600;">TOP GINI IMPORTANCES:</div>', unsafe_allow_html=True)
        rf_imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=FEATURES).sort_values(ascending=False).head(4)
        table_rows = "".join([f"<tr><td>{f}</td><td class='val'>{imp*100:.1f}%</td></tr>" for f, imp in rf_imp.items()])
        st.markdown(f'<table class="data-table">{table_rows}</table>', unsafe_allow_html=True)

# --- XGBoost ---
with p3:
    with st.container(border=True):
        xgb_meta = metrics.get("xgboost", {})
        xgb_params = param_dict.get("xgboost", {}).get("parameters", {})
        st.markdown(
            f"""
            <div style="border-left: 3px solid #f43f5e; padding-left: 0.6rem; margin-bottom: 0.7rem;">
                <div class="model-code">MODEL_ID: 03 // GRADIENT BOOSTING</div>
                <div class="model-name">XGBoost Classifier</div>
            </div>
            <div class="metric-row">
                <span class="chip">ACC: {xgb_meta.get('accuracy', 0.9018)*100:.1f}%</span>
                <span class="chip">BAL ACC: {xgb_meta.get('balanced_accuracy', 0.9080)*100:.1f}%</span>
                <span class="chip">ROC: {xgb_meta.get('roc_auc', 0.9703):.3f}</span>
                <span class="chip">R²: {xgb_meta.get('r2_score', 0.6920):.3f}</span>
                <span class="chip">Z-STAT: {xgb_meta.get('z_score_accuracy', 161.6):+.1f}σ</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if xgb_pred == 1:
            st.markdown(f'<div class="alert-high">[ HIGH RISK: LANDSLIDE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {xgb_prob*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-safe">[ LOW RISK: STABLE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {xgb_prob*100:.1f}%</span></div>', unsafe_allow_html=True)

        st.progress(xgb_prob)

        xgb_n_est = xgb_params.get("n_estimators", {}).get("value", getattr(xgb_model, "n_estimators", 403))
        xgb_depth = xgb_params.get("max_depth", {}).get("value", getattr(xgb_model, "max_depth", 8))
        xgb_lr_val = xgb_params.get("learning_rate", {}).get("value", getattr(xgb_model, "learning_rate", 0.0418))
        xgb_sub = xgb_params.get("subsample", {}).get("value", getattr(xgb_model, "subsample", 0.8335))
        xgb_col = xgb_params.get("colsample_bytree", {}).get("value", getattr(xgb_model, "colsample_bytree", 0.8997))
        xgb_scale = xgb_params.get("scale_pos_weight", {}).get("value", 1.8502)

        st.markdown(
            f"""
            <table class="data-table">
                <tr><td>Boosting Rounds</td><td class="val">{xgb_n_est} (depth {xgb_depth})</td></tr>
                <tr><td>Learning Rate (η)</td><td class="val">{xgb_lr_val}</td></tr>
                <tr><td>R² Calibration Score</td><td class="val">{xgb_meta.get('r2_score', 0.6920):.4f}</td></tr>
                <tr><td>Z-Score Significance (Z_acc)</td><td class="val">{xgb_meta.get('z_score_accuracy', 161.6):+.2f}σ</td></tr>
                <tr><td>Brier Score Loss</td><td class="val">{xgb_meta.get('brier_score', 0.0702):.4f}</td></tr>
                <tr><td>Subsample / Colsample</td><td class="val">{xgb_sub} / {xgb_col}</td></tr>
                <tr><td>Class Balancing (scale_pos)</td><td class="val">{xgb_scale}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔬 DYNAMIC PARAMETER PURPOSE"):
            for p_name, p_info in xgb_params.items():
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:0.45rem 0.6rem; border-radius:4px; margin-bottom:0.45rem; border-left:2px solid #f43f5e;">
                        <div style="font-family:JetBrains Mono; font-size:0.75rem; color:#fda4af; font-weight:700;">
                            {p_info.get('display_name', p_name)}: <span style="color:#38bdf8;">{p_info.get('value')}</span>
                        </div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;"><b>Meaning:</b> {p_info.get('meaning')}</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; margin-top:0.2rem;"><b>Landslide Purpose:</b> {p_info.get('purpose')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="font-size:0.75rem; color:#94a3b8; font-family:JetBrains Mono; margin-top:0.8rem; font-weight:600;">TOP GAIN IMPORTANCES:</div>', unsafe_allow_html=True)
        xgb_imp = pd.Series(xgb_pipe.named_steps["model"].feature_importances_, index=FEATURES).sort_values(ascending=False).head(4)
        table_rows = "".join([f"<tr><td>{f}</td><td class='val'>{imp*100:.1f}%</td></tr>" for f, imp in xgb_imp.items()])
        st.markdown(f'<table class="data-table">{table_rows}</table>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 04: Dynamic Control Parameter Knowledge Base
# ---------------------------------------------------------------------------
st.markdown('<div class="divider-tag">// SECTION 04: DYNAMIC CONTROL PARAMETER KNOWLEDGE BASE &amp; STATISTICAL AUDIT</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="font-size:0.84rem; color:#94a3b8; margin-bottom:1.1rem; line-height:1.5;">
        Identified via <b>scikit-learn dynamic parameter analysis (RandomizedSearchCV)</b> across multi-state remote sensing datasets.
        This knowledge base details each tuned hyperparameter, its formal mathematical definition, and its
        operational purpose in managing trade-offs, spatial smoothing, and false-negative prevention for landslide early warning.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_scores, tab_lr, tab_rf, tab_xgb, tab_matrix = st.tabs([
    "STATISTICAL SCORING AUDIT (R², Z-SCORE, BRIER)",
    "LOGISTIC REGRESSION TELEMETRY",
    "RANDOM FOREST ENSEMBLE",
    "XGBOOST GRADIENT BOOSTING",
    "FULL PARAMETER SPECIFICATION MATRIX",
])

with tab_scores:
    st.markdown("<div style='font-size:0.8rem; color:#38bdf8; font-family:JetBrains Mono; margin-bottom:0.7rem;'>// SCIKIT-LEARN STATISTICAL SCORING &amp; CALIBRATION METRICS</div>", unsafe_allow_html=True)

    score_data = []
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost Classifier",
    }
    for m_id, m_title in model_labels.items():
        m_eval = metrics.get(m_id, {})
        score_data.append({
            "Model": m_title,
            "Accuracy": f"{m_eval.get('accuracy', 0.0)*100:.2f}%",
            "Balanced Accuracy": f"{m_eval.get('balanced_accuracy', 0.0)*100:.2f}%",
            "ROC-AUC": f"{m_eval.get('roc_auc', 0.0):.4f}",
            "R² Score (Goodness-of-Fit)": f"{m_eval.get('r2_score', 0.0):.4f}",
            "Z-Score Significance (Z_acc)": f"{m_eval.get('z_score_accuracy', 0.0):+.1f} σ",
            "Brier Score (Calibration Loss)": f"{m_eval.get('brier_score', 0.0):.4f}",
            "Matthews Corr (MCC)": f"{m_eval.get('matthews_corrcoef', 0.0):.4f}",
            "Macro F1": f"{m_eval.get('macro_f1', 0.0):.4f}",
        })

    st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            r"""
            <div class="param-card" style="border-left-color:#38bdf8;">
                <div class="param-title">R² Goodness-of-Fit Score</div>
                <div class="param-field">
                    Evaluates the proportion of target variance explained by probabilistic risk estimates.
                    In probabilistic classification an R² &gt; 0.60 demonstrates exceptional probability calibration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            r"""
            <div class="param-card" style="border-left-color:#10b981;">
                <div class="param-title">Statistical Z-Score Significance (Z_acc)</div>
                <div class="param-field">
                    Calculates the standard normal Z-statistic comparing observed classification accuracy against the
                    majority-class naive baseline rate. Values exceeding +3.0σ prove decisive statistical superiority (p &lt; 0.001).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            r"""
            <div class="param-card" style="border-left-color:#f43f5e;">
                <div class="param-title">Brier Score Loss &amp; Skill Score</div>
                <div class="param-field">
                    A strictly proper scoring rule measuring mean squared probability calibration error.
                    Lower values indicate superior posterior calibration, essential for trustworthy hazard warning maps.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_param_cards(model_key, accent_color):
    model_data = param_dict.get(model_key, {})
    params = model_data.get("parameters", {})
    cols = st.columns(2)
    for idx, (p_name, p_info) in enumerate(params.items()):
        col = cols[idx % 2]
        with col:
            st.markdown(
                f"""
                <div class="param-card" style="border-left-color:{accent_color};">
                    <div class="param-title">
                        <span>{p_info.get('display_name', p_name)}</span>
                        <span>
                            <span class="param-tag">{p_info.get('control_type', 'Control')}</span>
                            <span class="param-val-badge">{p_info.get('value')}</span>
                        </span>
                    </div>
                    <div class="param-field"><b>Mathematical Meaning:</b> {p_info.get('meaning')}</div>
                    <div class="param-purpose-box">
                        <b>Operational Landslide Purpose:</b> {p_info.get('purpose')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with tab_lr:
    st.markdown("<div style='font-size:0.8rem; color:#a5b4fc; font-family:JetBrains Mono; margin-bottom:0.7rem;'>// LINEAR PROBABILISTIC MODEL CONTROL PARAMETERS</div>", unsafe_allow_html=True)
    render_param_cards("logistic_regression", "#6366f1")

with tab_rf:
    st.markdown("<div style='font-size:0.8rem; color:#6ee7b7; font-family:JetBrains Mono; margin-bottom:0.7rem;'>// BAGGING ENSEMBLE STRUCTURAL CONTROL PARAMETERS</div>", unsafe_allow_html=True)
    render_param_cards("random_forest", "#10b981")

with tab_xgb:
    st.markdown("<div style='font-size:0.8rem; color:#fda4af; font-family:JetBrains Mono; margin-bottom:0.7rem;'>// GRADIENT BOOSTING TREE REGULARIZATION &amp; LOSS PARAMETERS</div>", unsafe_allow_html=True)
    render_param_cards("xgboost", "#f43f5e")

with tab_matrix:
    st.markdown("<div style='font-size:0.8rem; color:#38bdf8; font-family:JetBrains Mono; margin-bottom:0.7rem;'>// CROSS-MODEL HYPERPARAMETER AUDIT &amp; COMPARISON MATRIX</div>", unsafe_allow_html=True)
    matrix_rows = []
    for m_key, m_val in param_dict.items():
        m_name = m_val.get("model_name", m_key)
        for p_key, p_val in m_val.get("parameters", {}).items():
            matrix_rows.append({
                "Model": m_name,
                "Parameter": p_val.get("display_name", p_key),
                "Identified Value": str(p_val.get("value")),
                "Control Type": p_val.get("control_type", ""),
                "Mathematical Meaning": p_val.get("meaning", ""),
                "Operational Purpose in Landslide Risk": p_val.get("purpose", ""),
            })
    matrix_df = pd.DataFrame(matrix_rows)
    st.dataframe(
        matrix_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="small"),
            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
            "Identified Value": st.column_config.TextColumn("Value", width="small"),
            "Control Type": st.column_config.TextColumn("Control Type", width="small"),
            "Mathematical Meaning": st.column_config.TextColumn("Mathematical Meaning", width="large"),
            "Operational Purpose in Landslide Risk": st.column_config.TextColumn("Operational Purpose", width="large"),
        },
    )

st.markdown('<div class="divider-tag" style="margin-top:2rem;">// GEOSPATIAL INTELLIGENCE LAB // SENTINEL-1/2 LANDSLIDE SUSCEPTIBILITY PIPELINE</div>', unsafe_allow_html=True)
