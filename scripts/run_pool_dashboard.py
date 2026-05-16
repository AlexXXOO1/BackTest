# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "ui" / "pool_dashboard.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_PATH)]))
