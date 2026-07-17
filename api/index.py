"""Vercel ASGI entrypoint.

Vercel discovers this module as a Python Serverless Function.  The rewrite in
``vercel.json`` passes both the public dashboard and API paths to FastAPI.
"""
from app.main import app

__all__ = ["app"]
