#!/usr/bin/env bash
# ml_snapshot.sh — paste output to Claude to continue ML mentoring
# Run from: ~/ml-engineering/ml-foundation

echo "===== ML ENVIRONMENT SNAPSHOT ====="
echo "Date: $(date)"
echo ""

echo "--- SYSTEM ---"
uname -a
echo ""

echo "--- PYTHON ---"
python --version 2>&1
which python
echo ""

echo "--- UV / VENV ---"
uv --version 2>/dev/null || echo "uv not found"
echo "Active venv: $VIRTUAL_ENV"
echo ""

echo "--- INSTALLED PACKAGES ---"
pip list 2>/dev/null | head -60
echo ""

echo "--- PROJECT STRUCTURE ---"
tree . --gitignore 2>/dev/null || find . -not -path './.git/*' -not -path './.venv/*' | sort
echo ""

echo "--- pyproject.toml ---"
cat pyproject.toml 2>/dev/null || echo "not found"
echo ""

echo "--- SRC FILES ---"
find src -type f 2>/dev/null | while read f; do
  echo ""
  echo ">> $f"
  cat "$f"
done
echo ""

echo "--- NOTEBOOKS (names only) ---"
find notebooks -name "*.ipynb" 2>/dev/null | sort
echo ""

echo "--- NOTEBOOK CELL SUMMARY (01_numpy_foundation.ipynb) ---"
python3 - <<'PYEOF' 2>/dev/null
import json, sys, pathlib
nb_path = pathlib.Path("notebooks/01_numpy_foundation.ipynb")
if not nb_path.exists():
    print("Notebook not found")
    sys.exit()
nb = json.loads(nb_path.read_text())
cells = nb.get("cells", [])
print(f"Total cells: {len(cells)}")
for i, cell in enumerate(cells):
    ctype = cell.get("cell_type","")
    src = "".join(cell.get("source",""))[:120].replace("\n"," ")
    print(f"  [{i+1}] {ctype}: {src}")
PYEOF
echo ""

echo "--- GIT LOG (last 10 commits) ---"
git log --oneline -10 2>/dev/null || echo "No git history"
echo ""

echo "===== END SNAPSHOT ====="
