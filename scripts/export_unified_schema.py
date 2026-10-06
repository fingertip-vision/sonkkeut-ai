"""Export the same typed HTTP contract without loading or downloading weights."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonkkeut_ai.api import create_app

destination = ROOT / "docs/unified-ai.openapi.json"
destination.write_text(json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
