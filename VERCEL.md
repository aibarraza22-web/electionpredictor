# Vercel deployment

This repository now includes `api/index.py`, the Vercel ASGI entrypoint, and a catch-all rewrite in `vercel.json`. This is required because Vercel only exposes Python serverless functions below `api/` by default; without the rewrite, visiting `/` can produce a Vercel `404: NOT_FOUND` page.

## Deploy

1. Import the repository in Vercel with the project root set to this repository.
2. Use the default build settings; Vercel detects `api/index.py` and installs `requirements.txt`.
3. Deploy. Both `/` and `/api/...` are rewritten internally to the FastAPI application.
4. Open `/api/data-health` after deployment. It must return JSON with `"mode": "demo"` before sharing the URL.

The current deployment intentionally shows synthetic demo records only. It must not be represented as a live 2026 forecast. Configure a managed database and vetted timestamped source adapters before enabling any live mode.

## Troubleshooting

* **`404: NOT_FOUND` at `/`:** verify that the deployment includes `vercel.json` and `api/index.py`, and that Vercel’s root directory is not set to a child directory.
* **Function error:** inspect Vercel Function Logs; verify Python dependencies from `requirements.txt` installed successfully.
* **Stale demo database:** serverless filesystems are ephemeral. For production, replace SQLite with PostgreSQL and run migrations before traffic.
