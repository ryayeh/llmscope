from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.providers.huggingface_provider import collect_huggingface_local_diagnostics


def main() -> int:
    diagnostics = collect_huggingface_local_diagnostics()
    print(json.dumps(diagnostics.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
