"""
Patch app.py: insert _render_feature_help() calls after each slider line
by targeting exact line numbers.
"""

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Map of substring to identify each slider line -> feature key to inject after it
SLIDER_PATCHES = {
    'inputs["elevation"] = st.slider': "elevation",
    'inputs["slope"] = st.slider': "slope",
    'inputs["spi"] = st.slider': "spi",
    'inputs["tri"] = st.slider': "tri",
    'inputs["curvature"] = st.slider': "curvature",
    'inputs["NDVI"] = st.slider': "NDVI",
    'inputs["EVI"] = st.slider': "EVI",
    'inputs["GRVI"] = st.slider': "GRVI",
    'inputs["mNDWI"] = st.slider': "mNDWI",
    'inputs["GLCMMean"] = st.slider': "GLCMMean",
    'inputs["GLCMCorrelation"] = st.slider': "GLCMCorrelation",
    'inputs["twi"] = st.slider': "twi",
    'inputs["aspect"] = st.slider': "aspect",
}

new_lines = []
for line in lines:
    new_lines.append(line)
    for substr, key in SLIDER_PATCHES.items():
        if substr in line:
            # Determine indentation of slider line
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * indent + f'_render_feature_help("{key}")\n')
            break  # Only one match per line

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Patched. Total lines now: {len(new_lines)}")
