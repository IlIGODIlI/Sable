"""
Inject FEATURE_INFO dict and rewrite the slider section in app.py
to add a dropdown expander under each slider.
"""

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# ── 1. Insert FEATURE_INFO after DEFAULT_VALUES block ──────────────────────
FEATURE_INFO_BLOCK = '''

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

'''

# Insert FEATURE_INFO just after DEFAULT_VALUES closing brace
target = "}\n\n\n@st.cache_resource"
content = content.replace(target, "}\n" + FEATURE_INFO_BLOCK + "\n\n@st.cache_resource", 1)

# ── 2. Rewrite the slider section ──────────────────────────────────────────
# Helper to build the expander HTML for one feature key
OLD_SLIDER_SECTION = '''    inputs["elevation"] = st.slider("Elevation (m)", -150.0, 1500.0, float(inputs["elevation"]), 1.0)
    inputs["slope"] = st.slider("Slope Gradient (degrees)", 0.0, 60.0, float(inputs["slope"]), 0.5)
    inputs["spi"] = st.slider("Stream Power Index (SPI)", -15.0, 15.0, float(inputs["spi"]), 0.1)
    inputs["tri"] = st.slider("Terrain Ruggedness Index (TRI)", 0.0, 8.0, float(inputs["tri"]), 0.05)
    inputs["curvature"] = st.slider("Surface Curvature", -0.050, 0.050, float(inputs["curvature"]), 0.001, format="%.4f")

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
    inputs["EVI"] = st.slider("EVI (Enhanced Vegetation)", -10.0, 10.0, float(inputs["EVI"]), 0.1)
    inputs["GRVI"] = st.slider("GRVI (Green-Red Vegetation)", -0.50, 0.50, float(inputs["GRVI"]), 0.01)
    inputs["mNDWI"] = st.slider("mNDWI (Water Index)", -0.80, 0.80, float(inputs["mNDWI"]), 0.01)

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
    inputs["GLCMCorrelation"] = st.slider("GLCM Correlation", -0.50, 1.00, float(inputs["GLCMCorrelation"]), 0.01)
    inputs["twi"] = st.slider("Topographic Wetness (TWI)", 0.0, 15.0, float(inputs["twi"]), 0.1)
    inputs["aspect"] = st.slider("Terrain Aspect Orientation (\\u00b0)", 0.0, 360.0, float(inputs["aspect"]), 1.0)'''

NEW_SLIDER_SECTION = '''    inputs["elevation"] = st.slider("Elevation (m)", -150.0, 1500.0, float(inputs["elevation"]), 1.0)
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
    inputs["aspect"] = st.slider("Terrain Aspect Orientation (\\u00b0)", 0.0, 360.0, float(inputs["aspect"]), 1.0)
    _render_feature_help("aspect")'''

content = content.replace(OLD_SLIDER_SECTION, NEW_SLIDER_SECTION, 1)

# ── 3. Insert _render_feature_help function before "render_param_help" ──────
HELPER_FN = '''
def _render_feature_help(key: str):
    """Render a compact dropdown expander with feature description and seasonal ranges."""
    info = FEATURE_INFO.get(key)
    if not info:
        return
    season_rows = "".join(
        f\'\'\'<tr>
            <td style="padding:0.25rem 0.5rem 0.25rem 0; color:#94a3b8; white-space:nowrap; font-weight:600;">{season}</td>
            <td style="padding:0.25rem 0; color:#cbd5e1;">{rng}</td>
        </tr>\'\'\'
        for season, rng in info["seasons"].items()
    )
    with st.expander(f"\\u2139\\ufe0f About: {info[\'label\']}"):
        st.markdown(
            f\'\'\'<div style="font-size:0.76rem; line-height:1.6; font-family:\'Inter\',sans-serif;">
  <div style="margin-bottom:0.5rem;">
    <span style="color:#38bdf8; font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; font-weight:700;">WHAT IT IS</span><br>
    <span style="color:#e2e8f0;">{info["meaning"]}</span>
  </div>
  <div style="margin-bottom:0.6rem;">
    <span style="color:#10b981; font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; font-weight:700;">ROLE IN LANDSLIDE RISK</span><br>
    <span style="color:#e2e8f0;">{info["role"]}</span>
  </div>
  <div>
    <span style="color:#f59e0b; font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; font-weight:700;">SEASONAL VALUE RANGES</span>
    <table style="width:100%; margin-top:0.35rem; border-collapse:collapse; font-size:0.73rem;">
      {season_rows}
    </table>
  </div>
</div>\'\'\',
            unsafe_allow_html=True,
        )

'''

# Insert before render_param_help
content = content.replace(
    "\ndef render_param_help():",
    HELPER_FN + "\ndef render_param_help():",
    1,
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done. Lines:", content.count("\\n"))
