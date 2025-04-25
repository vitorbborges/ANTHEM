import pandas as pd
import toml
from ydata_profiling import ProfileReport

# ─── Configuration ─────────────────────────────────────────────────────────
TOML_PATH = "documentation/feature_docs.toml"
PARQUET_PATH = "data/processed_data/S3-approx-coordinates.parquet"
OUTPUT_HTML = "documentation/feature_docs.html"

# ─── 1) Load data & metadata ───────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)
full_docs = toml.load(TOML_PATH)

# ─── 2) Prepare descriptions & subset ──────────────────────────────────────
variable_descriptions = {
    col: meta.get("description", "") for col, meta in full_docs.items()
}
df_sel = df[list(variable_descriptions.keys())]

# ─── 3) Generate ProfileReport ────────────────────────────────────────────
profile = ProfileReport(
    df_sel,
    title="🛠️ Feature Set Profiling",
    variables={"descriptions": variable_descriptions},
    correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},
    interactions={"continuous": True},
    missing_diagrams={"bar": True, "matrix": True},
)
profile.to_file(OUTPUT_HTML)

print(f"Written feature docs to {OUTPUT_HTML}")
