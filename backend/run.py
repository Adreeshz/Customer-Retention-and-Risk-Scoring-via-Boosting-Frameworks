# ==============================================================================
# SECTION: SERVICE ENTRY POINT — starts the FastAPI churn-scoring service on
# 127.0.0.1:8000 so the React frontend can connect via its REST endpoints
# ==============================================================================
import sys
from pathlib import Path

import uvicorn

# ==============================================================================
# SECTION: PATH BOOTSTRAP — allows "python run.py" from any working directory
# ==============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import app  # noqa: E402
from config import API_HOST, API_PORT  # noqa: E402

# ==============================================================================
# SECTION: SERVER LAUNCH — single-worker local service used by the dashboard
# ==============================================================================
if __name__ == "__main__":
    print(f"ChurnLens backend training on first request — serving at http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
