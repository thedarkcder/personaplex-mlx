"""Run the live server on a side port.

Useful for testing against a build while another instance owns port 8000:

    PERSONAPLEX_PORT=8123 .venv/bin/python tools/serve_side_port.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

import live_server_mlx as server  # noqa: E402

port = int(os.environ.get("PERSONAPLEX_PORT", "8123"))
host = os.environ.get("PERSONAPLEX_HOST", "127.0.0.1")
uvicorn.run(server.app, host=host, port=port, log_level="info")
