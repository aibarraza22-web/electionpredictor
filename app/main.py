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
def home(): return '''<!doctype html><html lang=en><head><meta name=viewport content="width=device-width,initial-scale=1"><title>Congressional Forecast Lab</title><style>body{font:16px system-ui;max-width:1180px;margin:auto;background:#09111f;color:#e7eefb;padding:2rem}.banner{background:#8b3b12;padding:1rem;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}.card,table{background:#15233a;padding:1rem;border-radius:8px}a{color:#7fcbff}table{width:100%;border-collapse:collapse;padding:0}th,td{text-align:left;padding:.65rem;border-bottom:1px solid #2b3b56}.bar{display:flex;justify-content:space-between;align-items:center;margin:1.5rem 0}select{padding:.4rem}</style></head><body><h1>Congressional Forecast Lab</h1><p class=banner><b>DEMO MODE — not live forecasting data.</b> Synthetic deterministic inputs demonstrate the product. No result is certain and no demo figure should be used as a live forecast.</p><div id=summary class=grid aria-live=polite><div class=card>Loading control simulations…</div></div><div class=bar><h2 id=title>Competitive House races</h2><label>Chamber <select id=chamber><option value=house>House</option><option value=senate>Senate</option></select></label></div><table><thead><tr><th>Race</th><th>Democratic win chance</th><th>Projected margin</th><th>Rating</th><th>Data quality</th></tr></thead><tbody id=races><tr><td colspan=5>Loading forecasts…</td></tr></tbody></table><p><a href=/docs>OpenAPI</a> · <a href=/api/research>Research Registry</a> · <a href=/api/models>Model Laboratory</a> · <a href=/api/backtests>Sequential backtests</a> · <a href=/api/data-health>Data health</a></p><script>const pct=x=>`${(100*x).toFixed(1)}%`;async function load(c='house'){let [f,ctl,h]=await Promise.all([fetch(`/api/forecast/${c}`).then(x=>x.json()),fetch('/api/forecast/control').then(x=>x.json()),fetch('/api/data-health').then(x=>x.json())]);summary.innerHTML=`<div class=card><h2>House control</h2><b>${pct(ctl.house.democratic_control_probability)}</b><br>Democratic probability</div><div class=card><h2>Senate control</h2><b>${pct(ctl.senate.democratic_control_probability)}</b><br>Democratic probability</div><div class=card><h2>Data health</h2><b>${h.mode}</b><br>${h.warning}</div>`;title.textContent=`Competitive ${c[0].toUpperCase()+c.slice(1)} races`;races.innerHTML=f.forecasts.sort((a,b)=>Math.abs(a.dem_probability-.5)-Math.abs(b.dem_probability-.5)).slice(0,25).map(r=>`<tr><td><a href="/api/races/${r.race_id}">${r.race_id}</a></td><td>${pct(r.dem_probability)}</td><td>${r.expected_margin.toFixed(1)} D</td><td>${r.rating}</td><td>${r.data_quality}</td></tr>`).join('')}</script><script>chamber.onchange=e=>load(e.target.value);load()</script></body></html>'''

# Static forecast routes must be registered before the dynamic chamber route.
# Otherwise FastAPI treats ``control`` as a chamber value and returns a 404.
@app.get('/api/forecast/control')
def control():
 return {'mode':'demo','house':simulate_control(fcs('house'),'house'),'senate':simulate_control(fcs('senate'),'senate',base_dem_seats=34)}
@app.get('/api/forecast/{chamber}')
def forecast(chamber:str):
 if chamber not in ('house','senate'): raise HTTPException(404)
 return {'mode':'demo','model_version':'2026.1','data_version':'demo-2026.1','forecasts':fcs(chamber)}
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
