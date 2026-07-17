import pytest

def test_api_coverage_and_admin_auth():
 fastapi=pytest.importorskip('fastapi')
 from fastapi.testclient import TestClient
 from app.main import app
 with TestClient(app) as c:
  assert len(c.get('/api/races?chamber=house').json())==435
  assert c.get('/api/forecast/control').status_code==200
  assert c.post('/api/admin/refresh').status_code==401
  assert c.post('/api/admin/refresh',headers={'Authorization':'Bearer demo-admin'}).status_code==200
