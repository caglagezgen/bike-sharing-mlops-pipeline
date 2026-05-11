import sys
from pathlib import Path

# Make sure the repo root is on sys.path so `import src.*` works when pytest
# is run from any directory without needing an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
