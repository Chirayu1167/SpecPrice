import os
import sys

# Make the project root importable so `import app` finds app.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: E402

# Vercel's Python runtime looks for a WSGI-compatible `app` object
