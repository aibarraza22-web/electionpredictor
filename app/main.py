from __future__ import annotations
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .store import init_db, races, build_forecasts, latest, conn, now
from .simulation import simulate_control

app=FastAPI(title='Congressional Forecast Lab',version='2026.1',description='Demo-mode, leakage-aware congressional forecasting API.')
@app.on_event('startup')
def startup(): init_db(); build_forecasts()
def fcs(chamber=None): return [x for x in build_forecasts() if not chamber or x['race_id'].startswith(chamber)]
@app.get('/',response_class=HTMLResponse)
def home(): return '''<!doctype html><title>Congressional Forecast Lab</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;background:#09111f;color:#e7eefb;padding:2rem}.banner{background:#8b3b12;padding:1rem;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}.card{background:#15233a;padding:1rem;border-radius:8px}a{color:#7fcbff}</style><h1>Congressional Forecast Lab</h1><p class=banner><b>DEMO MODE — not live forecasting data.</b> Synthetic deterministic inputs demonstrate the system; sources and timestamps are visible through the API.</p><div id=x class=grid></div><p><a href=/docs>OpenAPI</a> · <a href=/api/races?chamber=house>House races</a> · <a href=/api/research>Research Registry</a> · <a href=/api/models>Model Laboratory</a></p><script>Promise.all(['/api/forecast/control','/api/data-health']).then(x=>Promise.all(x.map(r=>r.json()))).then(([d,h])=>x.innerHTML=`<div class=card><h2>House</h2><b>${(d.house.democratic_control_probability*100).toFixed(1)}%</b> Democratic control</div><div class=card><h2>Senate</h2><b>${(d.senate.democratic_control_probability*100).toFixed(1)}%</b> Democratic control</div><div class=card><h2>Data health</h2>${h.mode}<br>${h.warning}</div>`)</script>'''
@app.get('/api/forecast/{chamber}')
def forecast(chamber:str):
 if chamber not in ('house','senate'): raise HTTPException(404)
 return {'mode':'demo','model_version':'2026.1','data_version':'demo-2026.1','forecasts':fcs(chamber)}
@app.get('/api/forecast/control')
def control():
 return {'mode':'demo','house':simulate_control(fcs('house'),'house'),'senate':simulate_control(fcs('senate'),'senate',base_dem_seats=34)}
@app.get('/api/races')
def list_races(chamber:str|None=None,state:str|None=None):
 return [r for r in races(chamber) if not state or r['state']==state]
@app.get('/api/races/{race_id}')
def race(race_id:str):
 r=next((r for r in races() if r['id']==race_id),None)
 if not r: raise HTTPException(404)
 return {**r,'forecast':next(x for x in fcs() if x['race_id']==race_id),'mode':'demo'}
@app.get('/api/races/{race_id}/history')
def history(race_id:str):
 c=conn(); result=[dict(x) for x in c.execute('select * from forecasts where race_id=? order by as_of',(race_id,))];c.close();return result
@app.get('/api/races/{race_id}/components')
def components(race_id:str):
 x=latest(race_id)
 if not x: raise HTTPException(404)
 import json; return {'race_id':race_id,'components':json.loads(x['components']),'as_of':x['as_of']}
@app.get('/api/races/{race_id}/{kind}')
def race_extra(race_id:str,kind:str):
 if kind not in ('polls','finance','comparables'): raise HTTPException(404)
 return {'race_id':race_id,kind:[], 'mode':'demo','note':'No fabricated source records are displayed.'}
@app.get('/api/models')
def models():
 c=conn();x=[dict(r) for r in c.execute('select * from model_versions')];c.close();return x
@app.get('/api/models/{model_id}')
def model(model_id:str): return {'id':model_id,'family':'chamber-specific regularized linear ensemble','status':'champion','formula_documentation':'METHODOLOGY.md'}
@app.get('/api/models/{model_id}/learning-history')
def learning(model_id:str): return {'model_id':model_id,'process':['freeze held-out forecast','score election','append completed cycle','re-estimate before next cycle'],'documentation':'SEQUENTIAL_LEARNING.md'}
@app.get('/api/research')
def research():
 c=conn();x=[dict(r) for r in c.execute('select * from research_claims')];c.close();return x
@app.get('/api/research/{claim_id}')
def claim(claim_id:str):
 c=conn();x=c.execute('select * from research_claims where id=?',(claim_id,)).fetchone();c.close()
 if not x:raise HTTPException(404)
 return dict(x)
@app.get('/api/metrics')
def metrics(): return {'metrics':['Brier score','log loss','margin MAE','margin RMSE','calibration','80/95 interval coverage'], 'status':'available through backtest runs'}
@app.get('/api/backtests')
def backtests(): return {'framework':'expanding-window prequential','midterms':['2006','2010','2014','2018','2022'],'warning':'Demo installation contains mechanics, not claimed historical performance.'}
@app.get('/api/backtests/{run_id}')
def backtest(run_id:str): return {'id':run_id,'status':'not_run','reason':'Import authoritative historical snapshots before publishing validation estimates.'}
class Scenario(BaseModel): national_environment:float=0
@app.post('/api/scenarios')
def scenario(s:Scenario): return {'label':'Scenario — not official forecast','national_environment':s.national_environment}
def admin(action:str, authorization:str|None, reason:str):
 if authorization!='Bearer demo-admin': raise HTTPException(401,'Admin bearer token required')
 c=conn();c.execute('insert into audit_logs(actor,action,reason,previous_value,new_value,created_at) values (?,?,?,?,?,?)',('demo-admin',action,reason,None,None,now()));c.commit();c.close();return {'accepted':True,'audit_logged':True,'action':action}
@app.post('/api/admin/{action}')
def admin_endpoint(action:str,reason:str='operator request',authorization:str|None=Header(None)):
 if action not in ('refresh','import','retrain','promote-model'):raise HTTPException(404)
 return admin(action,authorization,reason)
@app.get('/api/data-health')
def health(): return {'mode':'demo','warning':'Synthetic demo inputs; do not interpret as live forecasts.','latest_data_update':now(),'stale':False,'sources_configured':0}
