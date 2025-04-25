import pandas as pd
import toml
from ydata_profiling import ProfileReport

# ─── Configuration ─────────────────────────────────────────────────────────
TOML_PATH = "documentation/feature_docs.toml"
PARQUET_PATH = "data/processed_data/S1CO2-approx-coordinates.parquet"
OUTPUT_HTML = "documentation/feature_docs.html"

# ─── 1) Load data and metadata ─────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
full_docs = toml.load(TOML_PATH)

# ─── 2) Build a metadata DataFrame and turn it into HTML ───────────────────
meta_df = pd.DataFrame.from_dict(full_docs, orient="index")
meta_df.index.name = "column"
meta_df.reset_index(inplace=True)
# render with a CSS class for easier styling
meta_html = meta_df.to_html(index=False, border=1, classes="metadata-table")

# ─── 3) Prepare ProfileReport descriptions (just the human‐readable bit) ────
variable_descriptions = {
    col: meta.get("description", "") for col, meta in full_docs.items()
}

# ─── 4) Subset your DataFrame to documented columns ────────────────────────
df_sel = df[list(variable_descriptions.keys())]

# ─── 5) Generate the YData ProfileReport ───────────────────────────────────
profile = ProfileReport(
    df_sel,
    title="🛠️ Feature Set Profiling (Complete Dataset Docs + Profile)",
    variables={"descriptions": variable_descriptions},
    correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},
    interactions={"continuous": True},
    missing_diagrams={"bar": True, "matrix": True},
)
profile_html = profile.to_html()

# ─── 6) Merge dictionary and profile into one HTML with centered table ────
head, body_sep, rest = profile_html.partition("<body>")
body, body_end, tail = rest.partition("</body>")

combined_html = (
    head
    + """
    <style>
      /* Center the metadata table container */
      .metadata-container {
        max-width: 90%;
        margin: 20px auto;
        padding: 10px;
      }
      /* Optional: make the metadata table look nicer */
      .metadata-table {
        width: 100%;
        border-collapse: collapse;
      }
      .metadata-table th, .metadata-table td {
        padding: 8px;
        text-align: left;
      }
      .metadata-table th {
        background-color: #f0f0f0;
      }
    </style>
  """
    + body_sep
    + """
    <div class="metadata-container">
      <h1>📖 Dataset Dictionary</h1>
      {meta}
    </div>
    <hr/>
  """.format(
        meta=meta_html
    )
    + body
    + body_end
    + tail
)

# ─── 7) Write out the combined file ────────────────────────────────────────
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(combined_html)

print(f"Written combined documentation to {OUTPUT_HTML}")
