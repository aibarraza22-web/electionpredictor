"""Self-contained interactive dashboard.

One HTML document, zero external dependencies (CDN-free so it works behind
strict CSPs and on serverless). Reads everything from the JSON API, so the
page always reflects exactly what the pipeline stored — mode, provenance,
forecasts, backtests, baselines, and research registry included.

Palette: partisan semantic colors validated with the dataviz gates in both
light and dark modes (Dem #2a78d6/#3987e5, Rep #e34948/#e66767; neutral gray
midpoint for toss-ups).
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Congressional Forecast Lab</title>
<style>
:root{
  color-scheme: light;
  --surface:#fcfcfb; --panel:#f1f0ec; --panel2:#e8e6e0; --border:#d8d5cc;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a887f;
  --dem:#2a78d6; --rep:#e34948; --neutral:#8a887f;
  --dem-soft:rgba(42,120,214,.14); --rep-soft:rgba(227,73,72,.14);
  --live:#0f6b40; --demo:#a04a12; --warn:#8a6d00;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface:#12181f; --panel:#1a222c; --panel2:#212b37; --border:#2e3947;
    --ink:#eef3fa; --ink2:#b3bdca; --ink3:#7d8794;
    --dem:#3987e5; --rep:#e66767; --neutral:#8a94a1;
    --dem-soft:rgba(57,135,229,.18); --rep-soft:rgba(230,103,103,.18);
    --live:#35b57c; --demo:#e08040; --warn:#d4b85a;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface:#12181f; --panel:#1a222c; --panel2:#212b37; --border:#2e3947;
  --ink:#eef3fa; --ink2:#b3bdca; --ink3:#7d8794;
  --dem:#3987e5; --rep:#e66767; --neutral:#8a94a1;
  --dem-soft:rgba(57,135,229,.18); --rep-soft:rgba(230,103,103,.18);
  --live:#35b57c; --demo:#e08040; --warn:#d4b85a;
}
*{box-sizing:border-box}
body{font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:var(--surface);color:var(--ink)}
.wrap{max-width:1180px;margin:0 auto;padding:1.2rem 1.4rem 4rem}
header h1{font-size:1.45rem;margin:.2rem 0 .1rem}
header .sub{color:var(--ink2);font-size:.85rem}
.banner{border-radius:10px;padding:.7rem 1rem;margin:.9rem 0;font-size:.9rem;border:1px solid var(--border);background:var(--panel)}
.banner b.live{color:var(--live)} .banner b.demo{color:var(--demo)} .banner b.unconfigured{color:var(--warn)}
.banner small{color:var(--ink2)}
section{margin-top:1.6rem}
h2{font-size:1.02rem;margin:0 0 .6rem;letter-spacing:.01em}
h2 small{color:var(--ink3);font-weight:400}
.grid{display:grid;gap:.9rem}
.g2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding: .9rem 1rem}
.tile .big{font-size:1.9rem;font-weight:650;letter-spacing:-.01em}
.tile .lbl{color:var(--ink2);font-size:.8rem}
.tile .det{color:var(--ink3);font-size:.78rem;margin-top:.25rem}
.dem{color:var(--dem)} .rep{color:var(--rep)}
svg text{fill:var(--ink2);font-size:10px}
.axisline{stroke:var(--border)} .thresh{stroke:var(--ink3);stroke-dasharray:4 3}
.chips{display:flex;flex-wrap:wrap;gap:.45rem}
.chip{border:1px solid var(--border);background:var(--panel2);border-radius:999px;padding:.28rem .7rem;font-size:.8rem;cursor:pointer}
.chip:hover{border-color:var(--ink3)}
.chip .p{font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:.7rem}
.controls select,.controls input{background:var(--panel);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:.42rem .6rem;font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px}
th,td{padding:.5rem .65rem;text-align:left;white-space:nowrap}
thead th{background:var(--panel2);color:var(--ink2);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0}
tbody tr{border-top:1px solid var(--border);cursor:pointer}
tbody tr:hover{background:var(--panel)}
.rt{font-size:.75rem;border-radius:6px;padding:.12rem .45rem;font-weight:600;white-space:nowrap}
.rt.SD,.rt.LD,.rt.ND{background:var(--dem-soft);color:var(--dem)}
.rt.SR,.rt.LR,.rt.NR{background:var(--rep-soft);color:var(--rep)}
.rt.TU{background:var(--panel2);color:var(--ink2)}
.probbar{display:inline-block;width:110px;height:9px;border-radius:5px;background:var(--rep-soft);position:relative;vertical-align:middle}
.probbar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--dem-soft)}
.probbar b{position:absolute;top:-3px;width:2.5px;height:15px;background:var(--ink3);border-radius:2px}
.probbar em{position:absolute;top:-3px;width:2.5px;height:15px;border-radius:2px}
#detail{position:sticky;bottom:0;display:none;border-top:2px solid var(--border);background:var(--panel);max-height:62vh;overflow:auto;padding:1rem 1.2rem;border-radius:14px 14px 0 0;box-shadow:0 -8px 24px rgba(0,0,0,.22)}
#detail .close{float:right;cursor:pointer;border:1px solid var(--border);border-radius:8px;background:var(--panel2);color:var(--ink);padding:.2rem .6rem}
.comp{display:grid;grid-template-columns:170px 1fr 64px;gap:.4rem;align-items:center;font-size:.82rem}
.comp .bar{position:relative;height:12px;background:var(--panel2);border-radius:6px}
.comp .bar i{position:absolute;top:1px;bottom:1px;border-radius:5px}
.mono{font-variant-numeric:tabular-nums}
.muted{color:var(--ink3)} .small{font-size:.8rem}
.klist{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.6rem;font-size:.8rem}
.klist b{display:block;font-size:1.05rem}
.status{font-size:.72rem;border-radius:6px;padding:.1rem .4rem;background:var(--panel2);color:var(--ink2);font-weight:600}
.status.Production{color:var(--live)} .status.Experimental,.status.Collecting{color:var(--warn)}
#tip{position:fixed;pointer-events:none;display:none;background:var(--ink);color:var(--surface);font-size:.75rem;padding:.3rem .5rem;border-radius:6px;z-index:50}
a{color:var(--dem)}
footer{margin-top:2.2rem;color:var(--ink3);font-size:.78rem}
</style></head><body>
<div class="wrap">
<header>
  <h1>Congressional Forecast Lab</h1>
  <div class="sub">2026 U.S. House &amp; Senate forecast — every race, with uncertainty, provenance, and validated backtests.</div>
</header>
<div id="banner" class="banner">Loading…</div>

<section id="topline" style="display:none">
  <div class="grid g2">
    <div class="card"><h2>House control</h2><div id="houseTiles" class="grid g4"></div><div id="houseDist"></div></div>
    <div class="card"><h2>Senate control</h2><div id="senateTiles" class="grid g4"></div><div id="senateDist"></div></div>
  </div>
</section>

<section id="bgSec" style="display:none">
  <h2>Battlegrounds <small>— closest races by win probability among races with seat-level data (grade C+), derived from the model (not hand-picked)</small></h2>
  <p id="triage" class="small" style="color:var(--ink2);margin:.2rem 0 .6rem"></p>
  <div id="battle" class="chips"></div>
</section>

<section id="raceSec" style="display:none">
  <h2>Race explorer <small>— all races, click one for the deep dive</small></h2>
  <div class="controls">
    <select id="fChamber"><option value="">Both chambers</option><option value="house">House</option><option value="senate">Senate</option></select>
    <select id="fState"><option value="">All states</option></select>
    <select id="fRating"><option value="">All ratings</option></select>
    <input id="fSearch" placeholder="Search race or incumbent…">
    <span class="muted small" id="fCount" style="align-self:center"></span>
  </div>
  <div class="tablewrap" style="max-height:430px;overflow-y:auto"><table id="raceTable">
    <thead><tr>
      <th data-k="id">Race</th><th data-k="incumbent_name">Incumbent</th><th data-k="rating">Rating</th>
      <th data-k="dem_probability">Dem win %</th><th>P(D) ─ P(R)</th><th data-k="margin">Margin</th>
      <th data-k="low80">80% interval</th><th data-k="quality">Data grade</th>
    </tr></thead><tbody></tbody>
  </table></div>
</section>

<section id="modelSec" style="display:none">
  <h2>Model report card <small>— every number below was computed by a stored expanding-window backtest, never typed in</small></h2>
  <div class="grid g2" id="btCards"></div>
  <div class="card" style="margin-top:.9rem"><h2>Champion vs baselines <small>— identical walk-forward protocol; lower Brier/log loss is better</small></h2>
    <div class="tablewrap"><table id="cmpTable"><thead></thead><tbody></tbody></table></div>
    <div class="small muted" id="cmpNote" style="margin-top:.5rem"></div>
  </div>
</section>

<section id="researchSec" style="display:none">
  <h2>Research registry <small>— hypotheses with lifecycle status and the evidence behind each decision</small></h2>
  <div class="grid g2" id="claims"></div>
</section>

<footer id="foot"></footer>
</div>
<div id="detail"></div>
<div id="tip"></div>
<script>
"use strict";
const $=s=>document.querySelector(s), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=x=>(x*100).toFixed(1)+"%", sgn=x=>(x>0?"D+":"R+")+Math.abs(x).toFixed(1);
const RT={"Safe Democratic":"SD","Likely Democratic":"LD","Lean Democratic":"ND","Toss-up":"TU","Lean Republican":"NR","Likely Republican":"LR","Safe Republican":"SR"};
let RACES=[], FC={}, sortK="dem_probability", sortAsc=true;
const j=async u=>{const r=await fetch(u); if(!r.ok) throw new Error(u+" -> "+r.status); return r.json();};

function tip(ev,html){const t=$("#tip"); if(!html){t.style.display="none";return;}
  t.innerHTML=html; t.style.display="block";
  t.style.left=Math.min(ev.clientX+12,innerWidth-170)+"px"; t.style.top=(ev.clientY+14)+"px";}

function seatChart(dist, threshold, notUp){
  const entries=Object.entries(dist).map(([k,v])=>[+k+notUp,v]).sort((a,b)=>a[0]-b[0]);
  if(!entries.length) return "";
  const W=460,H=120,P=26, total=entries.reduce((s,e)=>s+e[1],0);
  const lo=entries[0][0], hi=entries[entries.length-1][0], span=Math.max(hi-lo,1);
  const bw=Math.max(1.5,(W-2*P)/(span+1)-1), mx=Math.max(...entries.map(e=>e[1]));
  let bars="";
  for(const [seats,n] of entries){
    const x=P+(seats-lo)/span*(W-2*P), h=(n/mx)*(H-38), col=seats>=threshold?"var(--dem)":"var(--rep)";
    bars+=`<rect x="${(x-bw/2).toFixed(1)}" y="${(H-24-h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5" fill="${col}" opacity=".85" data-t="${seats} Dem seats — ${(100*n/total).toFixed(1)}% of simulations"></rect>`;
  }
  const tx=P+(threshold-lo)/span*(W-2*P);
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:520px" role="img" aria-label="Democratic seat distribution">
    <line class="axisline" x1="${P}" y1="${H-24}" x2="${W-P}" y2="${H-24}"></line>${bars}
    <line class="thresh" x1="${tx}" y1="10" x2="${tx}" y2="${H-24}"></line>
    <text x="${tx+4}" y="18">majority ${threshold}</text>
    <text x="${P}" y="${H-8}">${lo}</text><text x="${W-P-18}" y="${H-8}">${hi}</text>
    <text x="${P}" y="10">Democratic seats — share of ${total.toLocaleString()} simulations</text></svg>`;
}

function tiles(c, chamber){
  const demP=c.democratic_control_probability, lead=demP>=.5;
  return `
  <div class="tile"><div class="lbl">Democratic control</div><div class="big ${lead?"dem":""}">${pct(demP)}</div></div>
  <div class="tile"><div class="lbl">Republican control</div><div class="big ${lead?"":"rep"}">${pct(1-demP)}</div></div>
  <div class="tile"><div class="lbl">Median Dem seats</div><div class="big mono">${c.median_democratic_seats}</div><div class="det">80%: ${c.interval_80[0]}–${c.interval_80[1]} · 95%: ${c.interval_95[0]}–${c.interval_95[1]}</div></div>
  <div class="tile"><div class="lbl">Tipping point</div><div class="big" style="font-size:1rem;padding-top:.35rem">${esc(c.tipping_point||"—")}</div><div class="det">${c.simulations.toLocaleString()} sims, shared national shock</div></div>`;
}

function probCell(f){
  const p=f.dem_probability;
  return `<span class="probbar"><i style="width:${(p*100).toFixed(1)}%"></i><b style="left:calc(50% - 1px)"></b><em style="left:calc(${(p*100).toFixed(1)}% - 1px);background:${p>=.5?"var(--dem)":"var(--rep)"}"></em></span>`;
}

function renderTable(){
  const ch=$("#fChamber").value, st=$("#fState").value, rt=$("#fRating").value, q=$("#fSearch").value.toLowerCase();
  let rows=RACES.filter(r=>{
    const f=FC[r.id]; if(!f) return false;
    return (!ch||r.chamber===ch)&&(!st||r.state===st)&&(!rt||f.rating===rt)&&
      (!q||r.id.toLowerCase().includes(q)||(r.incumbent_name||"").toLowerCase().includes(q)||(r.name||"").toLowerCase().includes(q));
  });
  rows.sort((a,b)=>{
    const fa=FC[a.id], fb=FC[b.id];
    const va=(sortK in fa)?fa[sortK]:a[sortK], vb=(sortK in fb)?fb[sortK]:b[sortK];
    const c=(typeof va==="number")?va-vb:String(va??"").localeCompare(String(vb??""));
    return sortAsc?c:-c;
  });
  $("#fCount").textContent=rows.length+" races";
  $("#raceTable tbody").innerHTML=rows.map(r=>{
    const f=FC[r.id];
    return `<tr data-id="${r.id}">
      <td>${esc(r.name||r.id)}${r.special?' <span class="muted small">(special)</span>':""}${r.open_seat?' <span class="muted small">(open)</span>':""}</td>
      <td>${esc(r.incumbent_name||"—")}${r.incumbent_party?` <span class="muted">(${r.incumbent_party})</span>`:""}</td>
      <td><span class="rt ${RT[f.rating]||"TU"}">${esc(f.rating)}</span></td>
      <td class="mono">${pct(f.dem_probability)}</td>
      <td>${probCell(f)}</td>
      <td class="mono">${sgn(f.margin)}</td>
      <td class="mono muted">${sgn(f.low80)} … ${sgn(f.high80)}</td>
      <td class="mono">${esc(f.quality)}</td></tr>`;
  }).join("");
}

async function openDetail(id){
  const r=RACES.find(x=>x.id===id), f=FC[id]; if(!r||!f) return;
  const d=$("#detail"); d.style.display="block";
  d.innerHTML=`<button class="close" onclick="this.parentElement.style.display='none'">close ✕</button>
    <h2 style="margin:0">${esc(r.name||id)} <span class="muted small">${r.chamber}${r.special?" · special election":""}${r.open_seat?" · open seat":""} · ${esc(r.election_system)}</span></h2>
    <div class="small muted">Incumbent: ${esc(r.incumbent_name||"none/vacant")} ${r.incumbent_party?"("+r.incumbent_party+")":""} · model ${esc(f.model_version)} · data ${esc(f.data_version)} · as of ${esc(f.as_of)}</div>
    <div class="grid g4" style="margin:.8rem 0">
      <div class="tile"><div class="lbl">Democratic win</div><div class="big ${f.dem_probability>=.5?"dem":""}">${pct(f.dem_probability)}</div></div>
      <div class="tile"><div class="lbl">Republican win</div><div class="big ${f.dem_probability<.5?"rep":""}">${pct(1-f.dem_probability)}</div></div>
      <div class="tile"><div class="lbl">Projected margin</div><div class="big mono" style="font-size:1.4rem">${sgn(f.margin)}</div><div class="det">80%: ${sgn(f.low80)} … ${sgn(f.high80)}<br>95%: ${sgn(f.low95)} … ${sgn(f.high95)}</div></div>
      <div class="tile"><div class="lbl">Rating / data grade</div><div style="padding-top:.3rem"><span class="rt ${RT[f.rating]||"TU"}">${esc(f.rating)}</span> <span class="status">grade ${esc(f.quality)}</span></div></div>
    </div>
    <div id="dComp"></div><div id="dModels" style="margin-top:.8rem"></div>
    <div id="dPolls" class="small" style="margin-top:.7rem">Loading polls…</div><div id="dHist" class="small muted" style="margin-top:.5rem"></div>`;
  const comp=JSON.parse(f.components||"{}"), tier=comp._model; delete comp._model;
  const entries=Object.entries(comp); const mx=Math.max(...entries.map(([,v])=>Math.abs(v)),1);
  $("#dComp").innerHTML=`<h2 style="margin-top:.4rem">Why this forecast <small>— additive margin components (points), ${tier==="full"?"polls + fundamentals tier":"fundamentals tier (no polls yet for this race)"}</small></h2>`+
    entries.map(([k,v])=>{
      const w=Math.abs(v)/mx*50;
      const left=v<0? (50-w):50;
      return `<div class="comp"><span>${esc(k)}</span><span class="bar"><i style="left:${left}%;width:${w}%;background:${v>=0?"var(--dem)":"var(--rep)"}"></i><b style="position:absolute;left:50%;top:0;bottom:0;width:1.5px;background:var(--ink3)"></b></span><span class="mono" style="text-align:right">${v>=0?"D+":"R+"}${Math.abs(v).toFixed(2)}</span></div>`;
    }).join("");
  d.scrollIntoView({behavior:"smooth",block:"end"});
  try{
    const mm=await j(`/api/races/${id}/models`);
    if(mm.models.length>1){
      $("#dModels").innerHTML=`<h2 style="margin:0 0 .4rem">Model comparison <small>— what each method says about this race; the champion drives the official forecast</small></h2>
      <div class="tablewrap"><table><thead><tr><th>Model</th><th>Dem win</th><th>Margin</th><th>80% interval</th><th>Rating</th></tr></thead><tbody>`+
      mm.models.map(x=>`<tr style="cursor:default"><td>${x.model_version===mm.champion?`<b>${esc(x.model_version)} (champion)</b>`:esc(x.model_version)}</td>
        <td class="mono">${pct(x.dem_probability)}</td><td class="mono">${sgn(x.margin)}</td>
        <td class="mono muted">${sgn(x.low80)} … ${sgn(x.high80)}</td>
        <td><span class="rt ${RT[x.rating]||"TU"}">${esc(x.rating)}</span></td></tr>`).join("")+`</tbody></table></div>`;
    }
  }catch(e){}
  try{
    const [polls,hist]=await Promise.all([j(`/api/races/${id}/polls`), j(`/api/races/${id}/history`)]);
    $("#dPolls").innerHTML= polls.polls.length
      ? `<b>${polls.polls.length} ingested polls.</b> Latest: `+polls.polls.slice(-3).reverse().map(p=>`${esc(p.pollster)} ${esc(p.poll_date)}: <span class="mono">${sgn(p.dem_margin)}</span>`).join(" · ")
      : `<span class="muted">${esc(polls.note||"No polls ingested for this race — the model widens uncertainty instead of assuming a tie.")}</span>`;
    $("#dHist").textContent="Frozen snapshots: "+hist.map(h=>`${h.as_of} (${(h.dem_probability*100).toFixed(1)}%)`).join(" → ");
  }catch(e){ $("#dPolls").textContent=""; }
}

function fmt(x,d=4){return x==null?"—":(+x).toFixed(d);}

async function main(){
  const h=await j("/api/data-health");
  const b=$("#banner");
  if(h.mode==="unconfigured"){
    b.innerHTML=`<b class="unconfigured">NOT CONFIGURED.</b> No data ingested and no forecasts exist yet. Run the pipeline (see DEPLOYMENT.md), then reload. <small>${esc(h.warnings.join(" "))}</small>`;
    return;
  }
  b.innerHTML= h.mode==="live"
    ? `<b class="live">LIVE FORECAST</b> — built from ingested primary sources as of <b>${esc(h.last_forecast_as_of)}</b> · data ${esc(h.data_version)} · ${h.counts.election_results.toLocaleString()} results · ${h.counts.polls.toLocaleString()} polls ${h.warnings.length?`<br><small>${esc(h.warnings.join(" "))}</small>`:""}`
    : `<b class="demo">DEMO MODE — synthetic data, not a live forecast.</b> <small>${esc(h.warnings.join(" "))}</small>`;

  const [ctl,races,fh,fs]=await Promise.all([j("/api/forecast/control"),j("/api/races"),j("/api/forecast/house"),j("/api/forecast/senate")]);
  RACES=races; for(const f of fh.forecasts.concat(fs.forecasts)) FC[f.race_id]=f;

  $("#topline").style.display="block";
  $("#houseTiles").innerHTML=tiles(ctl.house,"house");
  $("#senateTiles").innerHTML=tiles(ctl.senate,"senate");
  $("#houseDist").innerHTML=seatChart(ctl.house.distribution,218,0);
  $("#senateDist").innerHTML=seatChart(ctl.senate.distribution,51,0);
  document.querySelectorAll("#houseDist rect,#senateDist rect").forEach(r=>{
    r.addEventListener("mousemove",e=>tip(e,r.dataset.t)); r.addEventListener("mouseleave",()=>tip(null));});

  // Grade-D races (no ingested seat history) share one fundamentals
  // prediction near 50% and would flood this list without race-specific
  // signal, so battlegrounds require seat-level data (grade C or better).
  const battle=RACES.map(r=>({r,f:FC[r.id]})).filter(x=>x.f&&x.f.quality<"D")
    .sort((a,b)=>Math.abs(a.f.dem_probability-.5)-Math.abs(b.f.dem_probability-.5)).slice(0,14);
  $("#bgSec").style.display="block";
  const all=Object.values(FC);
  const competitive=all.filter(f=>["Toss-up","Lean Democratic","Lean Republican"].includes(f.rating)).length;
  $("#triage").innerHTML=`<b>${competitive}</b> of ${all.length} races are competitive (Lean or Toss-up) — that is where research, polling, and candidate attention pay off. The other <b>${all.length-competitive}</b> are rated Safe/Likely by the model and need only monitoring.`;
  $("#battle").innerHTML=battle.map(({r,f})=>`<span class="chip" data-id="${r.id}">${esc(r.name||r.id)} <span class="p ${f.dem_probability>=.5?"dem":"rep"}">${pct(f.dem_probability)}</span></span>`).join("");
  $("#battle").addEventListener("click",e=>{const c=e.target.closest(".chip"); if(c) openDetail(c.dataset.id);});

  $("#raceSec").style.display="block";
  const states=[...new Set(RACES.map(r=>r.state))].sort();
  $("#fState").innerHTML+=states.map(s=>`<option>${s}</option>`).join("");
  const ratings=[...new Set(Object.values(FC).map(f=>f.rating))];
  const order=["Safe Democratic","Likely Democratic","Lean Democratic","Toss-up","Lean Republican","Likely Republican","Safe Republican"];
  $("#fRating").innerHTML+=order.filter(r=>ratings.includes(r)).map(r=>`<option>${r}</option>`).join("");
  for(const id of ["fChamber","fState","fRating"]) $("#"+id).addEventListener("change",renderTable);
  $("#fSearch").addEventListener("input",renderTable);
  document.querySelectorAll("#raceTable thead th[data-k]").forEach(th=>th.addEventListener("click",()=>{
    const k=th.dataset.k; if(sortK===k) sortAsc=!sortAsc; else {sortK=k; sortAsc=true;} renderTable();}));
  $("#raceTable tbody").addEventListener("click",e=>{const tr=e.target.closest("tr"); if(tr) openDetail(tr.dataset.id);});
  renderTable();

  try{
    const bt=await j("/api/backtests");
    const champs=bt.runs.filter(r=>!String(r.model_version).startsWith("baseline"));
    const seen={}, latest=[];
    for(const r of champs){ if(!seen[r.chamber]){seen[r.chamber]=1; latest.push(r);} }
    if(latest.length){
      $("#modelSec").style.display="block";
      $("#btCards").innerHTML=latest.map(r=>`<div class="card"><h2>${r.chamber} — held-out cycles ${r.cycles[0]}–${r.cycles[r.cycles.length-1]} <small>(${r.n_races} races)</small></h2>
        <div class="klist">
          <div><span class="muted">Brier</span><b class="mono">${fmt(r.brier)}</b></div>
          <div><span class="muted">Log loss</span><b class="mono">${fmt(r.log_loss)}</b></div>
          <div><span class="muted">Winner acc.</span><b class="mono">${pct(r.winner_accuracy)}</b></div>
          <div><span class="muted">Margin MAE</span><b class="mono">${fmt(r.margin_mae,2)} pts</b></div>
          <div><span class="muted">80% coverage</span><b class="mono">${pct(r.coverage80)}</b></div>
          <div><span class="muted">95% coverage</span><b class="mono">${pct(r.coverage95)}</b></div>
        </div>
        <div class="small muted" style="margin-top:.5rem">Calibration (forecast → observed): ${r.calibration.map(c=>`${c.bin}: ${(c.forecast*100).toFixed(0)}→${(c.observed*100).toFixed(0)} (n=${c.n})`).join(" · ")}</div></div>`).join("");
    }
    const cmp=await j("/api/models/comparison");
    const chambers=Object.keys(cmp.chambers);
    const models=[...new Set(chambers.flatMap(c=>Object.keys(cmp.chambers[c])))];
    models.sort((a,b)=>(a===cmp.champion?-1:b===cmp.champion?1:a.localeCompare(b)));
    $("#cmpTable thead").innerHTML="<tr><th>Model</th>"+chambers.map(c=>`<th>${c} Brier</th><th>${c} log loss</th><th>${c} acc.</th><th>${c} MAE</th>`).join("")+"</tr>";
    $("#cmpTable tbody").innerHTML=models.map(m=>{
      const cells=chambers.map(c=>{const x=cmp.chambers[c][m]; return x?`<td class="mono">${fmt(x.brier)}</td><td class="mono">${fmt(x.log_loss)}</td><td class="mono">${pct(x.winner_accuracy)}</td><td class="mono">${fmt(x.margin_mae,2)}</td>`:"<td>—</td><td>—</td><td>—</td><td>—</td>;}).join("");
      return `<tr style="cursor:default"><td>${m===cmp.champion?`<b>${esc(m)} (champion)</b>`:esc(m)}</td>${cells}</tr>`;}).join("");
    $("#cmpNote").textContent=cmp.note;
  }catch(e){ /* comparisons appear after the first pipeline run */ }

  try{
    const claims=await j("/api/research");
    if(claims.length){
      $("#researchSec").style.display="block";
      $("#claims").innerHTML=claims.map(c=>`<div class="card"><div style="display:flex;justify-content:space-between;gap:.5rem"><b>${esc(c.id)}</b><span class="status ${esc((c.status||"").split(" ")[0])}">${esc(c.status)}</span></div>
        <div style="margin:.3rem 0">${esc(c.claim)}</div>
        <div class="small muted">Metric: ${esc(c.metric)} · Mechanism: ${esc(c.mechanism)}</div>
        <div class="small" style="margin-top:.3rem"><b>Evidence:</b> ${esc(c.validation)}</div>
        <div class="small muted" style="margin-top:.2rem"><b>Decision:</b> ${esc(c.decision)}</div></div>`).join("");
    }
  }catch(e){}

  $("#foot").innerHTML=`Sources: ${h.sources.map(s=>`${esc(s.source)} (${s.records.toLocaleString()} records, retrieved ${esc((s.last_retrieved_at||"").slice(0,10))})`).join(" · ")}
   · <a href="/docs">OpenAPI</a> · <a href="/api/data-health">data health</a> · storage: ${esc(h.database_backend)}${h.durable_storage?"":" (non-durable!)"}`;
}
main().catch(e=>{$("#banner").innerHTML=`<b class="unconfigured">Error loading forecast:</b> ${esc(e.message)}`;});
</script>
</body></html>"""
