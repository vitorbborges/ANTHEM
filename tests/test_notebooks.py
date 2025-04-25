import os
import sys
import nbformat
import pytest
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor

# Paths and settings
NOTEBOOKS_DIR = Path("notebooks")
SKIP_PREFIXES = ["draft_", "scratch_", "temp_"]
TIMEOUT = 600

# Use project root as working directory and for imports
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

# Collect notebooks, skipping drafts
notebooks = [
    nb
    for nb in NOTEBOOKS_DIR.glob("*.ipynb")
    if not any(nb.name.startswith(prefix) for prefix in SKIP_PREFIXES)
]


@pytest.mark.parametrize("notebook_path", notebooks)
def test_notebook_execution(notebook_path):
    """Ensure each notebook runs without errors."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(project_root)}})
    except FileNotFoundError as e:
        pytest.skip(f"Skipping '{notebook_path.name}' (missing data): {e}")
    except Exception as e:
        pytest.fail(f"Notebook '{notebook_path.name}' failed: {e}")
