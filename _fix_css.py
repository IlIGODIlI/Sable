"""Fix broken CSS attribute selectors in app.py — the closing ] was stripped."""
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix all three broken CSS attribute selectors
content = content.replace(
    '[class*="css" {',
    '[class*="css"] {'
)
content = content.replace(
    'div[data-testid="stVerticalBlockBorderWrapper" {',
    'div[data-testid="stVerticalBlockBorderWrapper"] {'
)
content = content.replace(
    'div[data-testid="stSlider" label {',
    'div[data-testid="stSlider"] label {'
)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("CSS selectors fixed.")
