"""Vercel serverless entrypoint (works under any framework preset).

vercel.json rewrites every path here; @vercel/python installs
requirements.txt from the repository root and serves the ASGI app.
"""
from app.main import app

__all__ = ["app"]
