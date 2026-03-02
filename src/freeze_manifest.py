import hashlib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_all_seasons_base.csv"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "base_manifest.txt"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

size = BASE_PATH.stat().st_size
digest = sha256_file(BASE_PATH)
ts = datetime.now().isoformat(timespec="seconds")

text = (
    f"timestamp: {ts}\n"
    f"file: {BASE_PATH}\n"
    f"bytes: {size}\n"
    f"sha256: {digest}\n"
)

MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.write_text(text, encoding="utf-8")

print(text)
print("Saved manifest to:", MANIFEST_PATH)
