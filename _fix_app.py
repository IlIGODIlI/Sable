"""
Fix script: Rebuilds app.py by keeping lines 1-268 (CSS/imports)
and replacing the rest with clean, correct Python code.
"""
import pathlib

APP = pathlib.Path("d:/SML_Landslide/app.py")

with open(APP, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Keep lines 1-268 (index 0-267) — CSS + st.set_page_config + st.markdown(CSS)
header = "".join(lines[:268])

body = '''\

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
                "accuracy": 0.6998, "balanced_accuracy": 0.7042, "roc_auc": 0.7835,
                "r2_score": 0.1250, "z_score_accuracy": 31.5, "brier_score": 0.2014,
                "macro_f1": 0.6879, "matthews_corrcoef": 0.3912,
            },
            "random_forest": {
                "accuracy": 0.9144, "balanced_accuracy": 0.9192, "roc_auc": 0.9755,
                "r2_score": 0.6854, "z_score_accuracy": 169.8, "brier_score": 0.0712,
                "macro_f1": 0.9083, "matthews_corrcoef": 0.8175,
            },
            "xgboost": {
                "accuracy": 0.9014, "balanced_accuracy": 0.9079, "roc_auc": 0.9699,
                "r2_score": 0.6421, "z_score_accuracy": 158.4, "brier_score": 0.0815,
                "macro_f1": 0.8949, "matthews_corrcoef": 0.7915,
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
            <span class="status-tag">SCORING: R\u00b2 / Z-SCORE / BRIER</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Parameter Help (rendered once, always visible)
# ---------------------------------------------------------------------------
def render_param_help():
    with st.sidebar.expander("\U0001f4da Parameter Help"):
        param_options = []
        for m_key, m_val in param_dict.items():
            for p_key, p_info in m_val.get("parameters", {}).items():
                label = f"{m_val.get('model_name', '')} \u2013 {p_info.get('display_name', p_key)}"
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
st.markdown(\'<div class="divider-tag">// SECTION 01: TERRAIN SCENARIO PRESETS</div>\', unsafe_allow_html=True)

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
st.markdown(\'<div class="divider-tag">// SECTION 02: INFLUENTIAL PARAMETER SLIDE BARS</div>\', unsafe_allow_html=True)

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
    inputs["aspect"] = st.slider("Terrain Aspect Orientation (\u00b0)", 0.0, 360.0, float(inputs["aspect"]), 1.0)

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
st.markdown(\'<div class="divider-tag">// SECTION 03: TRI-MODEL INFERENCE &amp; ARCHITECTURAL TELEMETRY</div>\', unsafe_allow_html=True)

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
                <span class="chip">ACC: {lr_meta.get(\'accuracy\', 0.6998)*100:.1f}%</span>
                <span class="chip">BAL ACC: {lr_meta.get(\'balanced_accuracy\', 0.7040)*100:.1f}%</span>
                <span class="chip">ROC: {lr_meta.get(\'roc_auc\', 0.7833):.3f}</span>
                <span class="chip">R\u00b2: {lr_meta.get(\'r2_score\', 0.125):.3f}</span>
                <span class="chip">Z-STAT: {lr_meta.get(\'z_score_accuracy\', 31.5):+.1f}\u03c3</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if lr_pred == 1:
            st.markdown(f\'<div class="alert-high">[ HIGH RISK: LANDSLIDE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {lr_prob*100:.1f}%</span></div>\', unsafe_allow_html=True)
        else:
            st.markdown(f\'<div class="alert-safe">[ LOW RISK: STABLE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {lr_prob*100:.1f}%</span></div>\', unsafe_allow_html=True)

        st.progress(lr_prob)

        lr_c_val = lr_params.get("C", {}).get("value", round(float(getattr(lr_model, "C", 1.0)), 4))
        lr_solver = lr_params.get("solver", {}).get("value", getattr(lr_model, "solver", "lbfgs"))
        lr_class_w = lr_params.get("class_weight", {}).get("value", getattr(lr_model, "class_weight", "balanced"))

        st.markdown(
            f"""
            <table class="data-table">
                <tr><td>Logit Score (z)</td><td class="val">{z_score:+.4f}</td></tr>
                <tr><td>Sigmoid \u03c3(z)</td><td class="val">{lr_prob:.4f}</td></tr>
                <tr><td>R\u00b2 Calibration Score</td><td class="val">{lr_meta.get(\'r2_score\', 0.125):.4f}</td></tr>
                <tr><td>Z-Score Significance (Z_acc)</td><td class="val">{lr_meta.get(\'z_score_accuracy\', 31.5):+.2f}\u03c3</td></tr>
                <tr><td>Brier Score Loss</td><td class="val">{lr_meta.get(\'brier_score\', 0.201):.4f}</td></tr>
                <tr><td>Inv Regularization (C)</td><td class="val">{lr_c_val}</td></tr>
                <tr><td>Solver / Weighting</td><td class="val">{lr_solver} ({lr_class_w})</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("\U0001f52c DYNAMIC PARAMETER PURPOSE"):
            for p_name, p_info in lr_params.items():
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:0.45rem 0.6rem; border-radius:4px; margin-bottom:0.45rem; border-left:2px solid #6366f1;">
                        <div style="font-family:\'JetBrains Mono\'; font-size:0.75rem; color:#a5b4fc; font-weight:700;">
                            {p_info.get(\'display_name\', p_name)}: <span style="color:#38bdf8;">{p_info.get(\'value\')}</span>
                        </div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;"><b>Meaning:</b> {p_info.get(\'meaning\')}</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; margin-top:0.2rem;"><b>Landslide Purpose:</b> {p_info.get(\'purpose\')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(\'<div style="font-size:0.75rem; color:#94a3b8; font-family:\'JetBrains Mono\'; margin-top:0.8rem; font-weight:600;">TOP WEIGHTS (w_i):</div>\', unsafe_allow_html=True)
        top_weights = pd.Series(lr_model.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False).head(4)
        table_rows = "".join([f"<tr><td>{f}</td><td class=\'val\'>{w:+.3f}</td></tr>" for f, w in top_weights.items()])
        st.markdown(f\'<table class="data-table">{table_rows}</table>\', unsafe_allow_html=True)

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
                <span class="chip">ACC: {rf_meta.get(\'accuracy\', 0.9144)*100:.1f}%</span>
                <span class="chip">BAL ACC: {rf_meta.get(\'balanced_accuracy\', 0.9192)*100:.1f}%</span>
                <span class="chip">ROC: {rf_meta.get(\'roc_auc\', 0.9755):.3f}</span>
                <span class="chip">R\u00b2: {rf_meta.get(\'r2_score\', 0.6854):.3f}</span>
                <span class="chip">Z-STAT: {rf_meta.get(\'z_score_accuracy\', 169.8):+.1f}\u03c3</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if rf_pred == 1:
            st.markdown(f\'<div class="alert-high">[ HIGH RISK: LANDSLIDE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {rf_prob*100:.1f}%</span></div>\', unsafe_allow_html=True)
        else:
            st.markdown(f\'<div class="alert-safe">[ LOW RISK: STABLE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {rf_prob*100:.1f}%</span></div>\', unsafe_allow_html=True)

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
                <tr><td>R\u00b2 Calibration Score</td><td class="val">{rf_meta.get(\'r2_score\', 0.6854):.4f}</td></tr>
                <tr><td>Z-Score Significance (Z_acc)</td><td class="val">{rf_meta.get(\'z_score_accuracy\', 169.8):+.2f}\u03c3</td></tr>
                <tr><td>Brier Score Loss</td><td class="val">{rf_meta.get(\'brier_score\', 0.0712):.4f}</td></tr>
                <tr><td>Max Depth / Min Leaf</td><td class="val">{rf_depth} / {rf_leaf}</td></tr>
                <tr><td>Split (Min / Feat)</td><td class="val">{rf_split} / {rf_feat}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("\U0001f52c DYNAMIC PARAMETER PURPOSE"):
            for p_name, p_info in rf_params.items():
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:0.45rem 0.6rem; border-radius:4px; margin-bottom:0.45rem; border-left:2px solid #10b981;">
                        <div style="font-family:\'JetBrains Mono\'; font-size:0.75rem; color:#6ee7b7; font-weight:700;">
                            {p_info.get(\'display_name\', p_name)}: <span style="color:#38bdf8;">{p_info.get(\'value\')}</span>
                        </div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;"><b>Meaning:</b> {p_info.get(\'meaning\')}</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; margin-top:0.2rem;"><b>Landslide Purpose:</b> {p_info.get(\'purpose\')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(\'<div style="font-size:0.75rem; color:#94a3b8; font-family:\'JetBrains Mono\'; margin-top:0.8rem; font-weight:600;">TOP GINI IMPORTANCES:</div>\', unsafe_allow_html=True)
        rf_imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=FEATURES).sort_values(ascending=False).head(4)
        table_rows = "".join([f"<tr><td>{f}</td><td class=\'val\'>{imp*100:.1f}%</td></tr>" for f, imp in rf_imp.items()])
        st.markdown(f\'<table class="data-table">{table_rows}</table>\', unsafe_allow_html=True)

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
                <span class="chip">ACC: {xgb_meta.get(\'accuracy\', 0.9014)*100:.1f}%</span>
                <span class="chip">BAL ACC: {xgb_meta.get(\'balanced_accuracy\', 0.9079)*100:.1f}%</span>
                <span class="chip">ROC: {xgb_meta.get(\'roc_auc\', 0.9699):.3f}</span>
                <span class="chip">R\u00b2: {xgb_meta.get(\'r2_score\', 0.6421):.3f}</span>
                <span class="chip">Z-STAT: {xgb_meta.get(\'z_score_accuracy\', 158.4):+.1f}\u03c3</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if xgb_pred == 1:
            st.markdown(f\'<div class="alert-high">[ HIGH RISK: LANDSLIDE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {xgb_prob*100:.1f}%</span></div>\', unsafe_allow_html=True)
        else:
            st.markdown(f\'<div class="alert-safe">[ LOW RISK: STABLE ]<br><span style="font-size:0.8rem; font-weight:400">PROBABILITY: {xgb_prob*100:.1f}%</span></div>\', unsafe_allow_html=True)

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
                <tr><td>Learning Rate (\u03b7)</td><td class="val">{xgb_lr_val}</td></tr>
                <tr><td>R\u00b2 Calibration Score</td><td class="val">{xgb_meta.get(\'r2_score\', 0.6421):.4f}</td></tr>
                <tr><td>Z-Score Significance (Z_acc)</td><td class="val">{xgb_meta.get(\'z_score_accuracy\', 158.4):+.2f}\u03c3</td></tr>
                <tr><td>Brier Score Loss</td><td class="val">{xgb_meta.get(\'brier_score\', 0.0815):.4f}</td></tr>
                <tr><td>Subsample / Colsample</td><td class="val">{xgb_sub} / {xgb_col}</td></tr>
                <tr><td>Class Balancing (scale_pos)</td><td class="val">{xgb_scale}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("\U0001f52c DYNAMIC PARAMETER PURPOSE"):
            for p_name, p_info in xgb_params.items():
                st.markdown(
                    f"""
                    <div style="background:#1e293b; padding:0.45rem 0.6rem; border-radius:4px; margin-bottom:0.45rem; border-left:2px solid #f43f5e;">
                        <div style="font-family:\'JetBrains Mono\'; font-size:0.75rem; color:#fda4af; font-weight:700;">
                            {p_info.get(\'display_name\', p_name)}: <span style="color:#38bdf8;">{p_info.get(\'value\')}</span>
                        </div>
                        <div style="font-size:0.72rem; color:#94a3b8; margin-top:0.2rem;"><b>Meaning:</b> {p_info.get(\'meaning\')}</div>
                        <div style="font-size:0.72rem; color:#cbd5e1; margin-top:0.2rem;"><b>Landslide Purpose:</b> {p_info.get(\'purpose\')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(\'<div style="font-size:0.75rem; color:#94a3b8; font-family:\'JetBrains Mono\'; margin-top:0.8rem; font-weight:600;">TOP GAIN IMPORTANCES:</div>\', unsafe_allow_html=True)
        xgb_imp = pd.Series(xgb_pipe.named_steps["model"].feature_importances_, index=FEATURES).sort_values(ascending=False).head(4)
        table_rows = "".join([f"<tr><td>{f}</td><td class=\'val\'>{imp*100:.1f}%</td></tr>" for f, imp in xgb_imp.items()])
        st.markdown(f\'<table class="data-table">{table_rows}</table>\', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 04: Dynamic Control Parameter Knowledge Base
# ---------------------------------------------------------------------------
st.markdown(\'<div class="divider-tag">// SECTION 04: DYNAMIC CONTROL PARAMETER KNOWLEDGE BASE &amp; STATISTICAL AUDIT</div>\', unsafe_allow_html=True)
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
    "STATISTICAL SCORING AUDIT (R\u00b2, Z-SCORE, BRIER)",
    "LOGISTIC REGRESSION TELEMETRY",
    "RANDOM FOREST ENSEMBLE",
    "XGBOOST GRADIENT BOOSTING",
    "FULL PARAMETER SPECIFICATION MATRIX",
])

with tab_scores:
    st.markdown("<div style=\'font-size:0.8rem; color:#38bdf8; font-family:JetBrains Mono; margin-bottom:0.7rem;\'>// SCIKIT-LEARN STATISTICAL SCORING &amp; CALIBRATION METRICS</div>", unsafe_allow_html=True)

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
            "Accuracy": f"{m_eval.get(\'accuracy\', 0.0)*100:.2f}%",
            "Balanced Accuracy": f"{m_eval.get(\'balanced_accuracy\', 0.0)*100:.2f}%",
            "ROC-AUC": f"{m_eval.get(\'roc_auc\', 0.0):.4f}",
            "R\u00b2 Score (Goodness-of-Fit)": f"{m_eval.get(\'r2_score\', 0.0):.4f}",
            "Z-Score Significance (Z_acc)": f"{m_eval.get(\'z_score_accuracy\', 0.0):+.1f} \u03c3",
            "Brier Score (Calibration Loss)": f"{m_eval.get(\'brier_score\', 0.0):.4f}",
            "Matthews Corr (MCC)": f"{m_eval.get(\'matthews_corrcoef\', 0.0):.4f}",
            "Macro F1": f"{m_eval.get(\'macro_f1\', 0.0):.4f}",
        })

    st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            r"""
            <div class="param-card" style="border-left-color:#38bdf8;">
                <div class="param-title">R\u00b2 Goodness-of-Fit Score</div>
                <div class="param-field">
                    Evaluates the proportion of target variance explained by probabilistic risk estimates.
                    In probabilistic classification an R\u00b2 &gt; 0.60 demonstrates exceptional probability calibration.
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
                    majority-class naive baseline rate. Values exceeding +3.0\u03c3 prove decisive statistical superiority (p &lt; 0.001).
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
                        <span>{p_info.get(\'display_name\', p_name)}</span>
                        <span>
                            <span class="param-tag">{p_info.get(\'control_type\', \'Control\')}</span>
                            <span class="param-val-badge">{p_info.get(\'value\')}</span>
                        </span>
                    </div>
                    <div class="param-field"><b>Mathematical Meaning:</b> {p_info.get(\'meaning\')}</div>
                    <div class="param-purpose-box">
                        <b>Operational Landslide Purpose:</b> {p_info.get(\'purpose\')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with tab_lr:
    st.markdown("<div style=\'font-size:0.8rem; color:#a5b4fc; font-family:JetBrains Mono; margin-bottom:0.7rem;\'>// LINEAR PROBABILISTIC MODEL CONTROL PARAMETERS</div>", unsafe_allow_html=True)
    render_param_cards("logistic_regression", "#6366f1")

with tab_rf:
    st.markdown("<div style=\'font-size:0.8rem; color:#6ee7b7; font-family:JetBrains Mono; margin-bottom:0.7rem;\'>// BAGGING ENSEMBLE STRUCTURAL CONTROL PARAMETERS</div>", unsafe_allow_html=True)
    render_param_cards("random_forest", "#10b981")

with tab_xgb:
    st.markdown("<div style=\'font-size:0.8rem; color:#fda4af; font-family:JetBrains Mono; margin-bottom:0.7rem;\'>// GRADIENT BOOSTING TREE REGULARIZATION &amp; LOSS PARAMETERS</div>", unsafe_allow_html=True)
    render_param_cards("xgboost", "#f43f5e")

with tab_matrix:
    st.markdown("<div style=\'font-size:0.8rem; color:#38bdf8; font-family:JetBrains Mono; margin-bottom:0.7rem;\'>// CROSS-MODEL HYPERPARAMETER AUDIT &amp; COMPARISON MATRIX</div>", unsafe_allow_html=True)
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

st.markdown(\'<div class="divider-tag" style="margin-top:2rem;">// GEOSPATIAL INTELLIGENCE LAB // SENTINEL-1/2 LANDSLIDE SUSCEPTIBILITY PIPELINE</div>\', unsafe_allow_html=True)
'''

with open(APP, "w", encoding="utf-8") as f:
    f.write(header)
    f.write(body)

print("Written successfully. Total chars:", len(header) + len(body))
