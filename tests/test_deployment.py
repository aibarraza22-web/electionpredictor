import json
from pathlib import Path

def test_vercel_routes_public_requests_to_fastapi_entrypoint():
    config = json.loads(Path('vercel.json').read_text())
    assert config['rewrites'] == [{'source': '/(.*)', 'destination': '/api/index.py'}]
    entrypoint = Path('api/index.py').read_text()
    assert 'from app.main import app' in entrypoint
