import pandas as pd
import toml

# ─── Configuration ─────────────────────────────────────────────────────────
TOML_PATH = "documentation/feature_docs.toml"
OUTPUT_HTML = "documentation/dataset_dictionary.html"

# ─── 1) Load metadata ─────────────────────────────────────────────────────
full_docs = toml.load(TOML_PATH)

# ─── 2) Build DataFrame & HTML ────────────────────────────────────────────
meta_df = pd.DataFrame.from_dict(full_docs, orient="index")
meta_df.index.name = "column"
meta_df.reset_index(inplace=True)
meta_html = meta_df.to_html(index=False, border=1, classes="metadata-table")

# ─── 3) Wrap in minimal HTML with styling ─────────────────────────────────
html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dataset Dictionary</title>
  <style>
    .metadata-container {{ max-width: 90%; margin: 20px auto; padding: 10px; }}
    .metadata-table {{ width: 100%; border-collapse: collapse; }}
    .metadata-table th, .metadata-table td {{ padding: 8px; text-align: left; }}
    .metadata-table th {{ background-color: #f0f0f0; }}
  </style>
</head>
<body>
  <div class="metadata-container">
    <h1>📖 Dataset Dictionary</h1>
    {meta_html}
  </div>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Written dataset dictionary to {OUTPUT_HTML}")
