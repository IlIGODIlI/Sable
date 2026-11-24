"""Fix embedded single-quotes in font-family CSS strings inside single-quoted st.markdown calls."""
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace  font-family:'JetBrains Mono';  with  font-family:JetBrains Mono;
content = content.replace("font-family:'JetBrains Mono';", "font-family:JetBrains Mono;")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done.")
