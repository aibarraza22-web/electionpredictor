from __future__ import annotations
import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone
from .domain import RaceFeatures, EnsembleModel

import os

DB_PATH = (
    Path("/tmp/forecast_lab.sqlite")
    if os.getenv("VERCEL")
    else Path(__file__).parents[1] / "data" / "forecast_lab.sqlite"
)

STATES=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
SENATE_2026=['AL','AK','AR','CO','DE','GA','ID','IL','IA','KS','KY','LA','ME','MA','MI','MN','MS','MT','NE','NH','NJ','NM','NC','OK','OR','RI','SC','SD','TN','TX','VA','WV','WY']

def conn():
    c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row; return c

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    c=conn(); c.executescript('''CREATE TABLE IF NOT EXISTS races (id TEXT PRIMARY KEY,chamber TEXT,state TEXT,district TEXT,name TEXT,baseline REAL,archetype TEXT,incumbent_party TEXT,open_seat INTEGER,election_system TEXT,updated_at TEXT);
    CREATE TABLE IF NOT EXISTS forecasts (id INTEGER PRIMARY KEY AUTOINCREMENT,race_id TEXT,as_of TEXT,model_version TEXT,data_version TEXT,dem_probability REAL,margin REAL,low80 REAL,high80 REAL,low95 REAL,high95 REAL,rating TEXT,quality TEXT,components TEXT,immutable INTEGER DEFAULT 1, UNIQUE(race_id,as_of,model_version));
    CREATE TABLE IF NOT EXISTS research_claims (id TEXT PRIMARY KEY,claim TEXT,chamber TEXT,metric TEXT,mechanism TEXT,status TEXT,validation TEXT,decision TEXT,source TEXT);
    CREATE TABLE IF NOT EXISTS audit_logs (id INTEGER PRIMARY KEY AUTOINCREMENT,actor TEXT,action TEXT,reason TEXT,previous_value TEXT,new_value TEXT,created_at TEXT);
    CREATE TABLE IF NOT EXISTS raw_sources (id INTEGER PRIMARY KEY AUTOINCREMENT,source TEXT,retrieved_at TEXT,available_at TEXT,payload TEXT);
    CREATE TABLE IF NOT EXISTS model_versions (id TEXT PRIMARY KEY,chamber TEXT,status TEXT,created_at TEXT,description TEXT);
    CREATE TABLE IF NOT EXISTS election_results (race_id TEXT,cycle INTEGER,dem_margin REAL,available_at TEXT);''')
    if not c.execute('select count(*) from races').fetchone()[0]:
      for n in range(1, 436):
       state=STATES[(n-1)%len(STATES)]; district=f'{state}-{((n-1)//len(STATES))+1:02d}'
       rid=f'house-2026-{state}-{n:03d}'; baseline=((n*11)%31)-15
       c.execute('insert into races values (?,?,?,?,?,?,?,?,?,?,?)',(rid,'house',state,district,district,baseline,'newly_redrawn' if n%37==0 else ('sparse_polling' if n%9==0 else 'general'),'D' if baseline>0 else 'R',int(n%11==0),'plurality',now()))
      for i,state in enumerate(SENATE_2026):
       c.execute('insert into races values (?,?,?,?,?,?,?,?,?,?,?)',(f'senate-2026-{state}','senate',state,None,f'{state} Senate',((i*9)%27)-13,'open_seat' if state in ['MI','NC'] else 'incumbent_defense','D' if i%2 else 'R',int(state in ['MI','NC']),'ranked_choice' if state in ['AK','ME'] else 'plurality',now()))
      claims=[('H-001','District presidential lean is a strong initial baseline.','house','district_presidential_margin','Partisan alignment','Production','Sequentially evaluated','Included in general model','Project research synthesis'),('H-002','District polling adds information closer to election day.','house','polling_margin_decay','Recent opinion measures candidate state','Experimental','Requires vintage backtest','Awaiting additional vintages','Project research synthesis'),('S-001','Senate races are more candidate-sensitive than House races.','senate','candidate_quality','Statewide personal brands','Production','Sequentially evaluated','Separate Senate coefficient','Project research synthesis'),('S-002','State-specific polling errors should be partially pooled.','senate','state_polling_error','Shared measurement conditions','Experimental','Requires expanded source history','Not promoted yet','Project research synthesis')]
      c.executemany('insert into research_claims values (?,?,?,?,?,?,?,?,?)',claims)
      c.execute('insert into model_versions values (?,?,?,?,?)',('2026.1','both','champion',now(),'Transparent chamber-specific ensemble; demo data only'))
      c.commit()
    c.close()

def now(): return datetime.now(timezone.utc).isoformat()
def races(chamber=None):
 c=conn(); q='select * from races'+(' where chamber=?' if chamber else ''); r=[dict(x) for x in c.execute(q, (chamber,) if chamber else ())]; c.close(); return r
def build_forecasts(as_of=None):
 as_of=as_of or now()[:10]; model=EnsembleModel(); c=conn(); out=[]
 for r in races():
  f=RaceFeatures(r['id'],r['chamber'],r['state'],r['baseline'], national_environment=-1.0, incumbent_dem=r['incumbent_party']=='D',open_seat=bool(r['open_seat']),candidate_edge=.3 if r['open_seat'] else 0,finance_edge=0,expert_edge=0,poll_count=0,days_to_election=110,archetype=r['archetype'],election_system=r['election_system'])
  fc=model.forecast(f); row={**fc.__dict__,'interval80':list(fc.interval80),'interval95':list(fc.interval95)}; out.append(row)
  c.execute('insert or ignore into forecasts(race_id,as_of,model_version,data_version,dem_probability,margin,low80,high80,low95,high95,rating,quality,components) values (?,?,?,?,?,?,?,?,?,?,?,?,?)',(fc.race_id,as_of,'2026.1','demo-2026.1',fc.dem_probability,fc.expected_margin,*fc.interval80,*fc.interval95,fc.rating,fc.data_quality,json.dumps(fc.components)))
 c.commit();c.close();return out
def latest(race_id):
 c=conn(); x=c.execute('select * from forecasts where race_id=? order by id desc limit 1',(race_id,)).fetchone();c.close();return dict(x) if x else None
