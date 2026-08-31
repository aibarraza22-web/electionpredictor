"""Self-contained interactive dashboard.

One HTML document, zero external dependencies (CDN-free so it works behind
strict CSPs and on serverless). Reads everything from the JSON API, so the
page always reflects exactly what the pipeline stored — mode, provenance,
forecasts, backtests and baselines included.

Palette: partisan semantic colors validated with the dataviz gates in both
light and dark modes (Dem #2a78d6/#3987e5, Rep #d63f3e/#e66767; neutral gray
midpoint for toss-ups). Every text/background pair used for small text meets
WCAG AA 4.5:1 in both themes — the previous muted grey (#8a887f on #fcfcfb)
was 3.1:1 and failed.

Accessibility contract, kept deliberately explicit because it is easy to
regress: one h1 and a sane heading order, landmarks on every region, a skip
link, visible focus rings on every interactive element, labelled form
controls, sortable columns as real buttons carrying aria-sort, table rows
reachable and activatable from the keyboard, the race panel as a modal
dialog that traps focus and restores it on close, live regions for the
status banner and result count, and honoured prefers-reduced-motion.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Congressional Forecast Lab — 2026 House &amp; Senate</title>
<meta name="description" content="2026 U.S. House and Senate forecast: every race with uncertainty, provenance and validated backtests.">
<style>
:root{
  color-scheme: light;
  --surface:#fbfaf8; --raised:#ffffff; --panel:#f4f2ee; --panel2:#eae7e1;
  --border:#dedad2; --border-strong:#c6c1b7;
  --ink:#16150f; --ink2:#4c4a44; --ink3:#6b6961;
  --dem:#2264bd; --rep:#c93a39; --neutral:#6b6961;
  --dem-ink:#1b4f95; --rep-ink:#a52d2c;
  --dem-soft:rgba(34,100,189,.12); --rep-soft:rgba(201,58,57,.12);
  --live:#0e6b3e; --demo:#8f430f; --warn:#725900;
  --focus:#0b57d0;
  --shadow:0 1px 2px rgba(20,18,12,.05), 0 6px 20px rgba(20,18,12,.06);
  --r:14px; --r-sm:9px;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface:#0e1319; --raised:#161d26; --panel:#161d26; --panel2:#1f2833;
    --border:#2a3542; --border-strong:#3b4859;
    --ink:#eef3f9; --ink2:#b8c2ce; --ink3:#93a0ae;
    --dem:#5b9df0; --rep:#f07a79; --neutral:#93a0ae;
    --dem-ink:#8bbcf7; --rep-ink:#f5a09f;
    --dem-soft:rgba(91,157,240,.16); --rep-soft:rgba(240,122,121,.16);
    --live:#4cc98d; --demo:#eb9553; --warn:#e0c56a;
    --focus:#8ab4f8;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 26px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface:#0e1319; --raised:#161d26; --panel:#161d26; --panel2:#1f2833;
  --border:#2a3542; --border-strong:#3b4859;
  --ink:#eef3f9; --ink2:#b8c2ce; --ink3:#93a0ae;
  --dem:#5b9df0; --rep:#f07a79; --neutral:#93a0ae;
  --dem-ink:#8bbcf7; --rep-ink:#f5a09f;
  --dem-soft:rgba(91,157,240,.16); --rep-soft:rgba(240,122,121,.16);
  --live:#4cc98d; --demo:#eb9553; --warn:#e0c56a;
  --focus:#8ab4f8;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 26px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Inter,sans-serif;
  margin:0;background:var(--surface);color:var(--ink);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
:where(a,button,select,input,summary,[tabindex]):focus-visible{
  outline:3px solid var(--focus);outline-offset:2px;border-radius:6px;
}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;
    transition-duration:.01ms!important;scroll-behavior:auto!important}
}
.skip{position:absolute;left:-9999px;top:0;background:var(--raised);color:var(--ink);
  padding:.7rem 1rem;border-radius:0 0 var(--r-sm) 0;z-index:100;font-weight:600}
.skip:focus{left:0}
.wrap{max-width:1200px;margin:0 auto;padding:0 1.25rem 5rem}

/* ---------- top bar ---------- */
.topbar{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--surface) 88%,transparent);
  backdrop-filter:saturate(1.6) blur(10px);border-bottom:1px solid var(--border)}
.topbar-in{max-width:1200px;margin:0 auto;padding:.7rem 1.25rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.brand{display:flex;align-items:baseline;gap:.6rem;margin-right:auto;min-width:0}
.brand h1{font-size:1.06rem;margin:0;font-weight:650;letter-spacing:-.011em;white-space:nowrap}
.brand .yr{color:var(--ink3);font-size:.82rem;white-space:nowrap}
.pill{display:inline-flex;align-items:center;gap:.4rem;font-size:.74rem;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;border-radius:999px;padding:.26rem .62rem;
  border:1px solid var(--border-strong);background:var(--panel);color:var(--ink2)}
.pill .dot{width:.5rem;height:.5rem;border-radius:50%;background:currentColor}
.pill.live{color:var(--live);border-color:color-mix(in srgb,var(--live) 45%,var(--border))}
.pill.demo{color:var(--demo);border-color:color-mix(in srgb,var(--demo) 45%,var(--border))}
.pill.unconfigured{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,var(--border))}
.tbtn{display:inline-flex;align-items:center;justify-content:center;gap:.4rem;min-height:2.25rem;
  padding:.4rem .75rem;font:inherit;font-size:.82rem;border-radius:999px;cursor:pointer;
  border:1px solid var(--border-strong);background:var(--panel);color:var(--ink)}
.tbtn:hover{background:var(--panel2)}

/* ---------- hero ---------- */
.lede{padding:1.5rem 0 .4rem;max-width:62ch}
.lede p{margin:.35rem 0 0;color:var(--ink2);font-size:.95rem}
.meta{margin:.9rem 0 0;padding:.7rem .9rem;border:1px solid var(--border);border-radius:var(--r-sm);
  background:var(--panel);color:var(--ink2);font-size:.84rem}
.meta b{color:var(--ink)}
.meta .warn{display:block;margin-top:.35rem;color:var(--warn)}

section{margin-top:2.1rem;scroll-margin-top:4.5rem}
.shead{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;margin:0 0 .8rem}
h2{font-size:1.12rem;margin:0;font-weight:650;letter-spacing:-.008em}
h3{font-size:.98rem;margin:0 0 .55rem;font-weight:650}
.shead p{margin:0;color:var(--ink3);font-size:.85rem;max-width:70ch}

.grid{display:grid;gap:1rem}
.g2{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.g4{grid-template-columns:repeat(auto-fit,minmax(158px,1fr))}
/* Control cards hold exactly four tiles; auto-fit turned that into 3+1 with a
   ragged gap, so they get an explicit 2x2 that collapses to one column. */
.g2x2{grid-template-columns:repeat(2,minmax(0,1fr))}
@media (max-width:420px){.g2x2{grid-template-columns:1fr}}
.card{background:var(--raised);border:1px solid var(--border);border-radius:var(--r);
  padding:1.05rem 1.15rem;box-shadow:var(--shadow)}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:var(--r-sm);padding:.75rem .85rem}
.tile .lbl{color:var(--ink2);font-size:.78rem;font-weight:600;letter-spacing:.01em}
.tile .big{font-size:1.85rem;font-weight:680;letter-spacing:-.022em;line-height:1.1;margin-top:.1rem}
.tile .det{color:var(--ink3);font-size:.775rem;margin-top:.3rem;line-height:1.4}
.dem{color:var(--dem-ink)} .rep{color:var(--rep-ink)}
.mono{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1}
.muted{color:var(--ink3)} .small{font-size:.82rem}

svg text{fill:var(--ink2);font-size:10px}
.axisline{stroke:var(--border-strong)} .thresh{stroke:var(--ink3);stroke-dasharray:4 3}

.chips{display:flex;flex-wrap:wrap;gap:.5rem}
.chip{display:inline-flex;align-items:center;gap:.45rem;min-height:2.25rem;
  border:1px solid var(--border-strong);background:var(--panel);color:var(--ink);
  border-radius:999px;padding:.35rem .8rem;font:inherit;font-size:.85rem;cursor:pointer}
.chip:hover{background:var(--panel2);border-color:var(--ink3)}
.chip .p{font-weight:670;font-variant-numeric:tabular-nums}

.controls{display:flex;flex-wrap:wrap;gap:.6rem;margin-bottom:.85rem;align-items:center}
.controls select,.controls input{background:var(--raised);color:var(--ink);
  border:1px solid var(--border-strong);border-radius:var(--r-sm);padding:.5rem .65rem;
  font:inherit;font-size:.88rem;min-height:2.4rem}
.controls input{min-width:15rem}
.vh{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;
  clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap;border:0}

.tablewrap{overflow:auto;border:1px solid var(--border);border-radius:var(--r);background:var(--raised)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:.875rem}
caption{text-align:left;padding:.6rem .8rem;color:var(--ink3);font-size:.82rem}
th,td{padding:.55rem .7rem;text-align:left;white-space:nowrap}
/* The odds bar duplicates the Dem-win percentage next to it, so it is the
   column to drop before the data grade scrolls out of view. */
@media (max-width:1340px){#raceTable th:nth-child(5),#raceTable td:nth-child(5){display:none}}
@media (max-width:900px){#raceTable th:nth-child(2),#raceTable td:nth-child(2){display:none}}
thead th{background:var(--panel2);color:var(--ink2);font-weight:650;position:sticky;top:0;z-index:1;
  border-bottom:1px solid var(--border)}
thead th button{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:.3rem;
  width:100%;font:inherit;font-weight:650;color:inherit}
thead th button::after{content:"";opacity:.35;font-size:.8em}
thead th[aria-sort="ascending"] button::after{content:"▲";opacity:1}
thead th[aria-sort="descending"] button::after{content:"▼";opacity:1}
tbody tr{border-top:1px solid var(--border)}
tbody tr[data-id]{cursor:pointer}
tbody tr[data-id]:hover{background:var(--panel)}
tbody tr[data-id]:focus-visible{outline:3px solid var(--focus);outline-offset:-3px}
tbody td{border-top:1px solid var(--border)}

.rt{display:inline-block;font-size:.76rem;border-radius:6px;padding:.16rem .5rem;font-weight:670;white-space:nowrap}
.rt.SD,.rt.LD,.rt.ND{background:var(--dem-soft);color:var(--dem-ink)}
.rt.SR,.rt.LR,.rt.NR{background:var(--rep-soft);color:var(--rep-ink)}
.rt.TU{background:var(--panel2);color:var(--ink2)}
.grade{display:inline-block;min-width:1.5rem;text-align:center;font-weight:670;font-size:.78rem;
  border-radius:5px;padding:.1rem .35rem;background:var(--panel2);color:var(--ink2)}
.grade.A,.grade.B{background:color-mix(in srgb,var(--live) 16%,transparent);color:var(--live)}
.grade.D,.grade.F{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}

.probbar{display:inline-block;width:112px;height:10px;border-radius:5px;background:var(--rep-soft);
  position:relative;vertical-align:middle;border:1px solid var(--border)}
.probbar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px 0 0 4px;background:var(--dem-soft)}
.probbar b{position:absolute;top:-3px;width:2px;height:14px;background:var(--ink3);border-radius:2px}
.probbar em{position:absolute;top:-3px;width:3px;height:14px;border-radius:2px}

.comp{display:grid;grid-template-columns:minmax(140px,190px) 1fr 74px;gap:.5rem;align-items:center;
  font-size:.84rem;padding:.16rem 0}
.comp .bar{position:relative;height:13px;background:var(--panel2);border-radius:7px}
.comp .bar i{position:absolute;top:1px;bottom:1px;border-radius:6px}
.klist{display:grid;grid-template-columns:repeat(auto-fit,minmax(124px,1fr));gap:.7rem;font-size:.82rem}
.klist>div{background:var(--panel);border:1px solid var(--border);border-radius:var(--r-sm);padding:.5rem .6rem}
.klist b{display:block;font-size:1.12rem;margin-top:.1rem}

/* ---------- race dialog ---------- */
.backdrop{position:fixed;inset:0;background:rgba(10,12,16,.45);z-index:60;display:none}
.backdrop[data-open="1"]{display:block}
#detail{position:fixed;z-index:61;left:50%;transform:translateX(-50%);bottom:0;width:min(1100px,100%);
  display:none;background:var(--raised);border:1px solid var(--border);border-bottom:0;
  max-height:86vh;overflow:auto;padding:1.1rem 1.3rem 1.6rem;border-radius:var(--r) var(--r) 0 0;
  box-shadow:0 -10px 40px rgba(0,0,0,.3)}
#detail[data-open="1"]{display:block}
.dhead{display:flex;align-items:flex-start;gap:1rem;position:sticky;top:-1.1rem;
  background:var(--raised);padding:.15rem 0 .6rem;margin:-.15rem 0 0;z-index:2}
.dhead .t{min-width:0}
.close{margin-left:auto;flex:none;min-height:2.3rem;border:1px solid var(--border-strong);border-radius:999px;
  background:var(--panel);color:var(--ink);padding:.4rem .85rem;font:inherit;font-size:.85rem;cursor:pointer}
.close:hover{background:var(--panel2)}

#tip{position:fixed;pointer-events:none;display:none;background:var(--ink);color:var(--surface);
  font-size:.78rem;padding:.35rem .55rem;border-radius:6px;z-index:70;font-variant-numeric:tabular-nums}
a{color:var(--dem-ink);text-underline-offset:2px}
footer{margin-top:2.6rem;padding-top:1.1rem;border-top:1px solid var(--border);
  color:var(--ink3);font-size:.8rem}
footer a{color:var(--ink2)}
</style></head><body>
<a class="skip" href="#main">Skip to forecast</a>

<div class="topbar">
  <div class="topbar-in">
    <div class="brand">
      <h1>Congressional Forecast Lab</h1>
      <span class="yr">2026 midterms</span>
    </div>
    <span id="modePill" class="pill"><span class="dot"></span><span id="modeText">loading</span></span>
    <button id="themeBtn" class="tbtn" type="button" aria-live="polite">Theme: system</button>
  </div>
</div>

<div class="wrap">
<main id="main">
  <div class="lede">
    <p>Every U.S. House and Senate race, with calibrated uncertainty, full source
    provenance, and accuracy measured by stored walk-forward backtests.</p>
  </div>
  <div id="banner" class="meta" role="status" aria-live="polite">Loading forecast…</div>

  <section id="topline" hidden aria-labelledby="toplineH">
    <div class="shead"><h2 id="toplineH">Chamber control</h2>
      <p>Correlated simulations of every seat, including a shared national-swing term.</p></div>
    <div class="grid g2">
      <div class="card"><h3>House</h3><div id="houseTiles" class="grid g2x2"></div><div id="houseDist"></div></div>
      <div class="card"><h3>Senate</h3><div id="senateTiles" class="grid g2x2"></div><div id="senateDist"></div></div>
    </div>
  </section>

  <section id="bgSec" hidden aria-labelledby="bgH">
    <div class="shead"><h2 id="bgH">Battlegrounds</h2>
      <p>Closest races by win probability among those with seat-level evidence — derived from the model, not hand-picked.</p></div>
    <p id="triage" class="small" style="color:var(--ink2);margin:0 0 .8rem"></p>
    <div id="battle" class="chips"></div>
  </section>

  <section id="raceSec" hidden aria-labelledby="raceH">
    <div class="shead"><h2 id="raceH">Race explorer</h2>
      <p>All races. Select a row for the full breakdown.</p></div>
    <div class="controls">
      <label class="vh" for="fChamber">Chamber</label>
      <select id="fChamber"><option value="">Both chambers</option><option value="house">House</option><option value="senate">Senate</option></select>
      <label class="vh" for="fState">State</label>
      <select id="fState"><option value="">All states</option></select>
      <label class="vh" for="fRating">Rating</label>
      <select id="fRating"><option value="">All ratings</option></select>
      <label class="vh" for="fSearch">Search races</label>
      <input id="fSearch" type="search" placeholder="Search race or incumbent…">
      <span class="muted small mono" id="fCount" role="status" aria-live="polite"></span>
    </div>
    <div class="tablewrap" style="max-height:min(60vh,520px)"><table id="raceTable">
      <caption class="vh">Every 2026 race with its rating, win probability, projected margin and data grade. Select a row to open its detail panel.</caption>
      <thead><tr>
        <th scope="col" data-k="name"><button type="button">Race</button></th>
        <th scope="col" data-k="incumbent_name"><button type="button">Incumbent</button></th>
        <th scope="col" data-k="rating"><button type="button">Rating</button></th>
        <th scope="col" data-k="dem_probability" aria-sort="ascending"><button type="button">Dem win</button></th>
        <th scope="col"><span aria-hidden="true">Odds</span><span class="vh">Win probability bar</span></th>
        <th scope="col" data-k="margin"><button type="button">Margin</button></th>
        <th scope="col"><span>80% range</span></th>
        <th scope="col" data-k="quality"><button type="button">Data grade</button></th>
      </tr></thead><tbody></tbody></table></div>
  </section>

  <section id="modelSec" hidden aria-labelledby="modelH">
    <div class="shead"><h2 id="modelH">Model report card</h2>
      <p>Every number here was computed by a stored expanding-window backtest — none is typed in.</p></div>
    <div class="grid g2" id="btCards"></div>
    <div class="card" style="margin-top:1rem">
      <h3>Champion vs baselines <span class="muted small" style="font-weight:400">— identical walk-forward protocol; lower Brier and log loss are better</span></h3>
      <div class="tablewrap"><table id="cmpTable"><thead></thead><tbody></tbody></table></div>
      <p class="small muted" id="cmpNote" style="margin:.6rem 0 0"></p>
    </div>
  </section>
</main>

<footer>
  <p>Forecasts are frozen, immutable snapshots. Methodology, the research registry
  and every source are available through the JSON API:
  <a href="/api/data-health">data health</a> ·
  <a href="/api/backtests">backtests</a> ·
  <a href="/api/research">research registry</a> ·
  <a href="/docs">API docs</a>.</p>
</footer>
</div>

<div id="backdrop" class="backdrop"></div>
<div id="detail" role="dialog" aria-modal="true" aria-labelledby="dTitle" tabindex="-1"></div>
<div id="tip" role="presentation" aria-hidden="true"></div>
<script>
"use strict";
const $=s=>document.querySelector(s), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=x=>(x*100).toFixed(1)+"%", sgn=x=>Math.abs(x)<.005?"Even":(x>0?"D+":"R+")+Math.abs(x).toFixed(1);
const RT={"Safe Democratic":"SD","Likely Democratic":"LD","Lean Democratic":"ND","Toss-up":"TU","Lean Republican":"NR","Likely Republican":"LR","Safe Republican":"SR"};
let RACES=[], FC={}, sortK="dem_probability", sortAsc=true, lastFocus=null;
const j=async u=>{const r=await fetch(u); if(!r.ok) throw new Error(u+" -> "+r.status); return r.json();};

/* ---- theme: system by default, remembered when the user chooses ---- */
(function theme(){
  const order=["system","light","dark"];
  let cur="system";
  try{ cur=localStorage.getItem("cfl-theme")||"system"; }catch(e){}
  const apply=()=>{
    if(cur==="system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme",cur);
    const b=$("#themeBtn"); if(b) b.textContent="Theme: "+cur;
  };
  apply();
  addEventListener("DOMContentLoaded",()=>{
    apply();
    $("#themeBtn").addEventListener("click",()=>{
      cur=order[(order.indexOf(cur)+1)%order.length];
      try{ localStorage.setItem("cfl-theme",cur); }catch(e){}
      apply();
    });
  });
})();

function tip(ev,html){const t=$("#tip"); if(!html){t.style.display="none";return;}
  t.innerHTML=html; t.style.display="block";
  t.style.left=Math.min(ev.clientX+12,innerWidth-190)+"px"; t.style.top=(ev.clientY+14)+"px";}

function seatChart(dist, threshold, notUp, chamber){
  const entries=Object.entries(dist).map(([k,v])=>[+k+notUp,v]).sort((a,b)=>a[0]-b[0]);
  if(!entries.length) return "";
  const W=460,H=124,P=26, total=entries.reduce((s,e)=>s+e[1],0);
  const lo=entries[0][0], hi=entries[entries.length-1][0], span=Math.max(hi-lo,1);
  const bw=Math.max(1.5,(W-2*P)/(span+1)-1), mx=Math.max(...entries.map(e=>e[1]));
  let bars="";
  for(const [seats,n] of entries){
    const x=P+(seats-lo)/span*(W-2*P), h=(n/mx)*(H-40), col=seats>=threshold?"var(--dem)":"var(--rep)";
    bars+=`<rect x="${(x-bw/2).toFixed(1)}" y="${(H-24-h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="1.5" fill="${col}" opacity=".9" data-t="${seats} Dem seats — ${(100*n/total).toFixed(1)}% of simulations"></rect>`;
  }
  const tx=P+(threshold-lo)/span*(W-2*P);
  const label=`Simulated Democratic ${chamber} seat totals, from ${lo} to ${hi} seats across ${total.toLocaleString()} simulations. A majority needs ${threshold}.`;
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:520px;margin-top:.6rem" role="img" aria-label="${esc(label)}">
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
  <div class="tile"><div class="lbl">Dem seats <span class="muted" style="font-weight:400">${esc(c.headline_basis||"most likely")}</span></div><div class="big mono">${c.headline_democratic_seats ?? c.most_likely_democratic_seats ?? c.median_democratic_seats}</div><div class="det">favored ${c.favored_democratic_seats ?? "—"} · most likely ${c.most_likely_democratic_seats ?? "—"} · median ${c.median_democratic_seats}<br>80%: ${c.interval_80[0]}–${c.interval_80[1]} · 95%: ${c.interval_95[0]}–${c.interval_95[1]}</div></div>
  <div class="tile"><div class="lbl">Tipping point</div><div class="big" style="font-size:1.02rem;padding-top:.4rem;line-height:1.25">${esc(prettyRace(c.tipping_point))}</div><div class="det">the seat that decides the majority</div></div>`;
}

// "2026-senate-ME" -> "Maine Senate"; "2026-house-MI-10" -> "Michigan 10th"
const STATE_NAMES={AL:"Alabama",AK:"Alaska",AZ:"Arizona",AR:"Arkansas",CA:"California",CO:"Colorado",CT:"Connecticut",DE:"Delaware",FL:"Florida",GA:"Georgia",HI:"Hawaii",ID:"Idaho",IL:"Illinois",IN:"Indiana",IA:"Iowa",KS:"Kansas",KY:"Kentucky",LA:"Louisiana",ME:"Maine",MD:"Maryland",MA:"Massachusetts",MI:"Michigan",MN:"Minnesota",MS:"Mississippi",MO:"Missouri",MT:"Montana",NE:"Nebraska",NV:"Nevada",NH:"New Hampshire",NJ:"New Jersey",NM:"New Mexico",NY:"New York",NC:"North Carolina",ND:"North Dakota",OH:"Ohio",OK:"Oklahoma",OR:"Oregon",PA:"Pennsylvania",RI:"Rhode Island",SC:"South Carolina",SD:"South Dakota",TN:"Tennessee",TX:"Texas",UT:"Utah",VT:"Vermont",VA:"Virginia",WA:"Washington",WV:"West Virginia",WI:"Wisconsin",WY:"Wyoming"};
function prettyRace(id){
  if(!id) return "—";
  const p=String(id).split("-");
  if(p.length<3) return id;
  const st=STATE_NAMES[p[2]]||p[2];
  const special=String(id).endsWith("-special")?" (special)":"";
  if(p[1]==="senate") return `${st} Senate${special}`;
  const d=p[3]?String(parseInt(p[3],10)):"";
  const suf=d.endsWith("1")&&d!=="11"?"st":d.endsWith("2")&&d!=="12"?"nd":d.endsWith("3")&&d!=="13"?"rd":"th";
  return d?`${st} ${d}${suf}`:st;
}

function probCell(f){
  const p=f.dem_probability;
  return `<span class="probbar" role="img" aria-label="${pct(p)} Democratic"><i style="width:${(p*100).toFixed(1)}%"></i><b style="left:calc(50% - 1px)"></b><em style="left:calc(${(p*100).toFixed(1)}% - 1.5px);background:${p>=.5?"var(--dem)":"var(--rep)"}"></em></span>`;
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
  document.querySelectorAll("#raceTable thead th[data-k]").forEach(th=>{
    th.setAttribute("aria-sort", th.dataset.k===sortK ? (sortAsc?"ascending":"descending") : "none");
  });
  $("#raceTable tbody").innerHTML=rows.map(r=>{
    const f=FC[r.id], name=r.name||r.id;
    return `<tr data-id="${r.id}" tabindex="0" aria-label="${esc(name)}, ${esc(f.rating)}, ${pct(f.dem_probability)} Democratic. Open details.">
      <td>${esc(name)}${r.special?' <span class="muted small">(special)</span>':""}${r.open_seat?' <span class="muted small">(open)</span>':""}</td>
      <td>${esc(r.incumbent_name||"—")}${r.incumbent_party?` <span class="muted">(${r.incumbent_party})</span>`:""}</td>
      <td><span class="rt ${RT[f.rating]||"TU"}">${esc(f.rating)}</span></td>
      <td class="mono">${pct(f.dem_probability)}</td>
      <td>${probCell(f)}</td>
      <td class="mono">${sgn(f.margin)}</td>
      <td class="mono muted">${sgn(f.low80)} … ${sgn(f.high80)}</td>
      <td><span class="grade ${esc(f.quality)}">${esc(f.quality)}</span></td></tr>`;
  }).join("");
}

function closeDetail(){
  $("#detail").removeAttribute("data-open");
  $("#backdrop").removeAttribute("data-open");
  if(lastFocus&&lastFocus.isConnected) lastFocus.focus();
}

async function openDetail(id){
  const r=RACES.find(x=>x.id===id), f=FC[id]; if(!r||!f) return;
  lastFocus=document.activeElement;
  const d=$("#detail");
  d.setAttribute("data-open","1"); $("#backdrop").setAttribute("data-open","1");
  d.innerHTML=`<div class="dhead"><div class="t">
      <h2 id="dTitle" style="margin:0">${esc(r.name||id)}</h2>
      <div class="small muted">${r.chamber}${r.special?" · special election":""}${r.open_seat?" · open seat":""} · ${esc(r.election_system)} · incumbent ${esc(r.incumbent_name||"none/vacant")} ${r.incumbent_party?"("+r.incumbent_party+")":""}</div>
      <div class="small muted">model ${esc(f.model_version)} · data ${esc(f.data_version)} · as of ${esc(f.as_of)}</div>
    </div><button class="close" type="button" id="dClose">Close</button></div>
    <div class="grid g4" style="margin:.5rem 0 .9rem">
      <div class="tile"><div class="lbl">Democratic win</div><div class="big ${f.dem_probability>=.5?"dem":""}">${pct(f.dem_probability)}</div></div>
      <div class="tile"><div class="lbl">Republican win</div><div class="big ${f.dem_probability<.5?"rep":""}">${pct(1-f.dem_probability)}</div></div>
      <div class="tile"><div class="lbl">Projected margin</div><div class="big mono" style="font-size:1.35rem">${sgn(f.margin)}</div><div class="det">80%: ${sgn(f.low80)} … ${sgn(f.high80)}<br>95%: ${sgn(f.low95)} … ${sgn(f.high95)}</div></div>
      <div class="tile"><div class="lbl">Rating / data grade</div><div style="padding-top:.4rem"><span class="rt ${RT[f.rating]||"TU"}">${esc(f.rating)}</span> <span class="grade ${esc(f.quality)}">${esc(f.quality)}</span></div></div>
    </div>
    <div id="dComp"></div><div id="dModels" style="margin-top:1rem"></div>
    <div id="dPolls" class="small" style="margin-top:.8rem">Loading polls…</div><div id="dHist" class="small muted" style="margin-top:.5rem"></div>`;
  $("#dClose").addEventListener("click",closeDetail);
  d.focus();
  const comp=JSON.parse(f.components||"{}"), tier=comp._model, analysis=comp._analysis||{};
  delete comp._model; delete comp._analysis;
  const entries=Object.entries(comp).filter(([,v])=>typeof v==="number"); const mx=Math.max(...entries.map(([,v])=>Math.abs(v)),1);
  $("#dComp").innerHTML=`<h3>Why this forecast <span class="muted small" style="font-weight:400">— additive margin components in points, ${tier==="full"?"polls + fundamentals tier":"fundamentals tier (no polls yet for this race)"}</span></h3>`+
    entries.map(([k,v])=>{
      const w=Math.abs(v)/mx*50;
      const left=v<0? (50-w):50;
      return `<div class="comp"><span>${esc(k)}</span><span class="bar"><i style="left:${left}%;width:${w}%;background:${v>=0?"var(--dem)":"var(--rep)"}"></i><b style="position:absolute;left:50%;top:0;bottom:0;width:1.5px;background:var(--ink3)"></b></span><span class="mono" style="text-align:right">${v>=0?"D+":"R+"}${Math.abs(v).toFixed(2)}</span></div>`;
    }).join("")+campaignAnalysis(analysis);
  try{
    const mm=await j(`/api/races/${id}/models`);
    if(mm.models.length>1){
      $("#dModels").innerHTML=`<h3>Model comparison <span class="muted small" style="font-weight:400">— what each method says; the champion drives the official forecast</span></h3>
      <div class="tablewrap"><table><thead><tr><th scope="col">Model</th><th scope="col">Dem win</th><th scope="col">Margin</th><th scope="col">80% interval</th><th scope="col">Rating</th></tr></thead><tbody>`+
      mm.models.map(x=>`<tr><td>${x.model_version===mm.champion?`<b>${esc(x.model_version)} (champion)</b>`:esc(x.model_version)}</td>
        <td class="mono">${pct(x.dem_probability)}</td><td class="mono">${sgn(x.margin)}</td>
        <td class="mono muted">${sgn(x.low80)} … ${sgn(x.high80)}</td>
        <td><span class="rt ${RT[x.rating]||"TU"}">${esc(x.rating)}</span></td></tr>`).join("")+`</tbody></table></div>`;
    }
  }catch(e){}
  try{
    const [polls,hist]=await Promise.all([j(`/api/races/${id}/polls`), j(`/api/races/${id}/history`)]);
    $("#dPolls").innerHTML= polls.polls.length
      ? `<b>${polls.polls.length} ingested poll${polls.polls.length===1?"":"s"}.</b> Latest: `
        +polls.polls.slice(-3).reverse().map(p=>`${esc(p.pollster)} ${esc(p.poll_date)}: <span class="mono">${sgn(p.dem_margin)}</span>`).join(" · ")
        +(polls.current===false&&polls.note?`<div class="muted" style="margin-top:.35rem">${esc(polls.note)}</div>`:"")
      : `<span class="muted">${esc(polls.note||"No polls ingested for this race — the model widens uncertainty instead of assuming a tie.")}</span>`;
    $("#dHist").textContent="Frozen snapshots: "+hist.map(h=>`${h.as_of} (${(h.dem_probability*100).toFixed(1)}%)`).join(" → ");
  }catch(e){ $("#dPolls").textContent=""; }
}

function expertRatings(er,ov){
  if(!er||!er.raters||!er.raters.length) return "";
  const rows=er.raters.slice().sort((x,y)=>String(y.rating_date).localeCompare(String(x.rating_date)));
  return `<h3 style="margin-top:1rem">What the handicappers say <span class="muted small" style="font-weight:400">— ${er.n_raters} published ratings, newest ${esc(er.newest_rating_date)}${er.age_days!=null?` (${er.age_days} days old)`:""}; each rater's own latest call</span></h3>
    <div class="tablewrap"><table><thead><tr><th scope="col">Rater</th><th scope="col">Rating</th><th scope="col">As of</th></tr></thead><tbody>`+
    rows.map(r=>`<tr><td>${esc(r.rater)}</td><td class="mono">${esc(r.rating)}</td><td class="mono muted">${esc(r.rating_date)}</td></tr>`).join("")+
    `</tbody></table></div>
    <p class="small muted" style="margin:.45rem 0 0">Consensus <span class="mono">${(+er.consensus).toFixed(2)}</span> on a −4 (Safe R) to +4 (Safe D) scale · rater disagreement <span class="mono">${(+er.disagreement).toFixed(2)}</span>. ${ov&&ov.applied?`Moves this forecast by <span class="mono">${sgn(ov.margin_shift)}</span> (fitted ${(+ov.slope_margin_points_per_rating_step).toFixed(2)} margin points per rating step, blended at ${(ov.blend_weight*100).toFixed(0)}%).`:`Not used to move this forecast: ${esc((ov&&ov.reason)||"no fitted overlay for this seat")}.`}</p>`;
}

function campaignAnalysis(a){
  if(!a||!Object.keys(a).length) return "";
  const v=a.victory_bands||{}, fin=(a.finance||{}).comparison, ch=a.change_since_previous, cad=a.campaign_adjustment_detail||{};
  const money=fin?`D/R receipts ${fin.dem_to_rep_receipts_ratio??"—"}× · cash ${fin.dem_to_rep_cash_ratio??"—"}× · capacity signal ${Math.abs(fin.descriptive_campaign_signal||0)<.005?"neutral":(fin.descriptive_campaign_signal>0?"D":"R")+" "+Math.abs(fin.descriptive_campaign_signal).toFixed(2)} · credibility ${pct(fin.signal_credibility||0)}`:
    "No comparable Democratic and Republican FEC vintages yet";
  const inputs=(cad.active_inputs||[]).length?(cad.active_inputs||[]).join(", "):"none";
  const ov=a.expert_rating_overlay||{}, er=a.expert_ratings||null;
  return `${expertRatings(er,ov)}<h3 style="margin-top:1rem">How the published margin is built</h3>
    <div class="grid g4">
      <div class="tile"><div class="lbl">Structural baseline</div><div class="big mono" style="font-size:1.12rem">${sgn(a.structural_baseline_margin||0)}</div></div>
      <div class="tile"><div class="lbl">Polling adjustment</div><div class="big mono" style="font-size:1.12rem">${sgn(a.polling_adjustment||0)}</div></div>
      <div class="tile"><div class="lbl">Expert ratings</div><div class="big mono" style="font-size:1.12rem">${sgn(a.expert_rating_adjustment||0)}</div><div class="det">${ov.applied?`weight ${(ov.blend_weight*100).toFixed(0)}% · ${esc(ov.stratum||"")}`:esc(ov.reason||"not applied")}</div></div>
      <div class="tile"><div class="lbl">Campaign adjustment</div><div class="big mono" style="font-size:1.12rem">${sgn(a.campaign_adjustment||0)}</div><div class="det">${esc(inputs)} · +${(+cad.added_sigma||0).toFixed(2)} uncertainty</div></div>
      <div class="tile"><div class="lbl">Change since last snapshot</div><div class="big mono" style="font-size:1.12rem">${ch?sgn(ch.margin_points):"—"}</div><div class="det">${ch?(ch.dem_probability_points>=0?"+":"")+ch.dem_probability_points.toFixed(2)+" probability points":"first snapshot"}</div></div>
    </div>
    <div class="klist" style="margin-top:.7rem">
      <div><span class="muted">D narrow (0–4)</span><b class="mono">${pct(v.dem_narrow_0_to_4||0)}</b></div>
      <div><span class="muted">D by 4+</span><b class="mono">${pct(v.dem_by_at_least_4||0)}</b></div>
      <div><span class="muted">D by 8+</span><b class="mono">${pct(v.dem_by_at_least_8||0)}</b></div>
      <div><span class="muted">R narrow (0–4)</span><b class="mono">${pct(v.rep_narrow_0_to_4||0)}</b></div>
      <div><span class="muted">R by 4+</span><b class="mono">${pct(v.rep_by_at_least_4||0)}</b></div>
      <div><span class="muted">R by 8+</span><b class="mono">${pct(v.rep_by_at_least_8||0)}</b></div>
    </div>
    <p class="small muted" style="margin:.6rem 0 0"><b>Campaign finance:</b> ${esc(money)}. Stage: ${esc((a.finance||{}).campaign_stage||"unknown")}. Poll absorption: ${pct(cad.poll_absorption_multiplier||0)}. Competitive-race multiplier: ${pct(cad.competitive_race_multiplier||0)}.</p>`;
}

function fmt(x,d=4){return x==null?"—":(+x).toFixed(d);}

async function main(){
  const h=await j("/api/data-health");
  const b=$("#banner"), pill=$("#modePill"), pillText=$("#modeText");
  pill.className="pill "+h.mode; pillText.textContent=h.mode==="live"?"live":h.mode;
  if(h.mode==="unconfigured"){
    b.innerHTML=`<b>Not configured.</b> No data ingested and no forecasts exist yet. Run the pipeline (see DEPLOYMENT.md), then reload.${h.warnings.length?`<span class="warn">${esc(h.warnings.join(" "))}</span>`:""}`;
    return;
  }
  const cov=h.coverage||{};
  const covBits=[];
  if(cov.races) covBits.push(`${cov.races} races`);
  if(cov.with_expert_ratings!=null) covBits.push(`${cov.with_expert_ratings} with expert ratings`);
  if(cov.with_polls!=null) covBits.push(`${cov.with_polls} with polls`);
  if(cov.competitive_races!=null) covBits.push(`${cov.competitive_races_grade_a_or_b ?? "—"}/${cov.competitive_races} competitive races at grade A–B`);
  b.innerHTML= h.mode==="live"
    ? `Built from ingested primary sources as of <b>${esc(h.last_forecast_as_of)}</b> · data <span class="mono">${esc(h.data_version)}</span><br>
       ${h.counts.election_results.toLocaleString()} results · ${h.counts.polls.toLocaleString()} polls${h.counts.race_ratings?` · ${h.counts.race_ratings.toLocaleString()} expert ratings`:""}${covBits.length?` · ${esc(covBits.join(" · "))}`:""}
       ${h.warnings.length?`<span class="warn">${esc(h.warnings.join(" "))}</span>`:""}`
    : `<b>Demo mode — synthetic data, not a live forecast.</b>${h.warnings.length?`<span class="warn">${esc(h.warnings.join(" "))}</span>`:""}`;

  const [ctl,races,fh,fs]=await Promise.all([j("/api/forecast/control"),j("/api/races"),j("/api/forecast/house"),j("/api/forecast/senate")]);
  RACES=races; for(const f of fh.forecasts.concat(fs.forecasts)) FC[f.race_id]=f;

  $("#topline").hidden=false;
  $("#houseTiles").innerHTML=tiles(ctl.house,"house");
  $("#senateTiles").innerHTML=tiles(ctl.senate,"senate");
  $("#houseDist").innerHTML=seatChart(ctl.house.distribution,218,0,"House");
  $("#senateDist").innerHTML=seatChart(ctl.senate.distribution,51,0,"Senate");
  document.querySelectorAll("#houseDist rect,#senateDist rect").forEach(r=>{
    r.addEventListener("mousemove",e=>tip(e,r.dataset.t)); r.addEventListener("mouseleave",()=>tip(null));});

  // Grade-D races (no ingested seat history) share one fundamentals
  // prediction near 50% and would flood this list without race-specific
  // signal, so battlegrounds require seat-level data (grade C or better).
  const battle=RACES.map(r=>({r,f:FC[r.id]})).filter(x=>x.f&&x.f.quality<"D")
    .sort((a,b)=>Math.abs(a.f.dem_probability-.5)-Math.abs(b.f.dem_probability-.5)).slice(0,14);
  $("#bgSec").hidden=false;
  const all=Object.values(FC);
  const competitive=all.filter(f=>["Toss-up","Lean Democratic","Lean Republican"].includes(f.rating)).length;
  $("#triage").innerHTML=`<b>${competitive}</b> of ${all.length} races are competitive (Lean or Toss-up) — that is where research, polling and candidate attention pay off. The other <b>${all.length-competitive}</b> are rated Safe or Likely and need only monitoring.`;
  $("#battle").innerHTML=battle.map(({r,f})=>`<button class="chip" type="button" data-id="${r.id}">${esc(r.name||r.id)} <span class="p ${f.dem_probability>=.5?"dem":"rep"}">${pct(f.dem_probability)}</span></button>`).join("");
  $("#battle").addEventListener("click",e=>{const c=e.target.closest(".chip"); if(c) openDetail(c.dataset.id);});

  $("#raceSec").hidden=false;
  const states=[...new Set(RACES.map(r=>r.state))].sort();
  $("#fState").innerHTML+=states.map(s=>`<option>${s}</option>`).join("");
  const ratings=[...new Set(Object.values(FC).map(f=>f.rating))];
  const order=["Safe Democratic","Likely Democratic","Lean Democratic","Toss-up","Lean Republican","Likely Republican","Safe Republican"];
  $("#fRating").innerHTML+=order.filter(r=>ratings.includes(r)).map(r=>`<option>${r}</option>`).join("");
  for(const id of ["fChamber","fState","fRating"]) $("#"+id).addEventListener("change",renderTable);
  $("#fSearch").addEventListener("input",renderTable);
  document.querySelectorAll("#raceTable thead th[data-k] button").forEach(btn=>btn.addEventListener("click",()=>{
    const k=btn.parentElement.dataset.k; if(sortK===k) sortAsc=!sortAsc; else {sortK=k; sortAsc=true;} renderTable();}));
  const body=$("#raceTable tbody");
  body.addEventListener("click",e=>{const tr=e.target.closest("tr[data-id]"); if(tr) openDetail(tr.dataset.id);});
  body.addEventListener("keydown",e=>{
    if(e.key!=="Enter"&&e.key!==" ") return;
    const tr=e.target.closest("tr[data-id]"); if(!tr) return;
    e.preventDefault(); openDetail(tr.dataset.id);
  });
  renderTable();

  // Escape closes the race dialog; the backdrop click does too.
  addEventListener("keydown",e=>{ if(e.key==="Escape"&&$("#detail").hasAttribute("data-open")) closeDetail(); });
  $("#backdrop").addEventListener("click",closeDetail);
  // Keep focus inside the dialog while it is open.
  $("#detail").addEventListener("keydown",e=>{
    if(e.key!=="Tab") return;
    const f=[...$("#detail").querySelectorAll('a[href],button,select,input,[tabindex]:not([tabindex="-1"])')]
      .filter(el=>el.offsetParent!==null);
    if(!f.length) return;
    const first=f[0], last=f[f.length-1], panel=$("#detail");
    // The panel itself takes focus on open (so a screen reader announces the
    // dialog and its title). It is tabindex="-1", so it is not in `f` — and
    // without treating it as the reverse boundary, one Shift+Tab straight
    // after opening escapes the modal onto the page behind the backdrop.
    const atStart=document.activeElement===first||document.activeElement===panel;
    if(e.shiftKey&&atStart){ e.preventDefault(); last.focus(); }
    else if(!e.shiftKey&&document.activeElement===last){ e.preventDefault(); first.focus(); }
  });

  try{
    const bt=await j("/api/backtests");
    const champs=bt.runs.filter(r=>!String(r.model_version).startsWith("baseline")&&!String(r.model_version).startsWith("ablation")&&!String(r.model_version).startsWith("challenger"));
    const seen={}, latest=[];
    for(const r of champs){ if(!seen[r.chamber]){seen[r.chamber]=1; latest.push(r);} }
    if(latest.length){
      $("#modelSec").hidden=false;
      $("#btCards").innerHTML=latest.map(r=>`<div class="card"><h3>${r.chamber} — held-out cycles ${r.cycles[0]}–${r.cycles[r.cycles.length-1]} <span class="muted small" style="font-weight:400">(${r.n_races} races)</span></h3>
        <div class="klist">
          <div><span class="muted">Brier</span><b class="mono">${fmt(r.brier)}</b></div>
          <div><span class="muted">Log loss</span><b class="mono">${fmt(r.log_loss)}</b></div>
          <div><span class="muted">Winner acc.</span><b class="mono">${pct(r.winner_accuracy)}</b></div>
          <div><span class="muted">Margin MAE</span><b class="mono">${fmt(r.margin_mae,2)} pts</b></div>
          <div><span class="muted">80% coverage</span><b class="mono">${pct(r.coverage80)}</b></div>
          <div><span class="muted">95% coverage</span><b class="mono">${pct(r.coverage95)}</b></div>
        </div></div>`).join("");
    }
    const cmp=await j("/api/models/comparison");
    const chambers=Object.keys(cmp.chambers);
    const models=[...new Set(chambers.flatMap(c=>Object.keys(cmp.chambers[c])))];
    models.sort((a,b)=>(a===cmp.champion?-1:b===cmp.champion?1:a.localeCompare(b)));
    $("#cmpTable thead").innerHTML="<tr><th scope='col'>Model</th>"+chambers.map(c=>`<th scope='col'>${c} Brier</th><th scope='col'>${c} log loss</th><th scope='col'>${c} acc.</th><th scope='col'>${c} MAE</th>`).join("")+"</tr>";
    $("#cmpTable tbody").innerHTML=models.map(m=>{
      const cells=chambers.map(c=>{const x=cmp.chambers[c][m]; return x?`<td class="mono">${fmt(x.brier)}</td><td class="mono">${fmt(x.log_loss)}</td><td class="mono">${pct(x.winner_accuracy)}</td><td class="mono">${fmt(x.margin_mae,2)}</td>`:"<td>—</td><td>—</td><td>—</td><td>—</td>";}).join("");
      return `<tr><td>${m===cmp.champion?`<b>${esc(m)} (champion)</b>`:esc(m)}</td>${cells}</tr>`;}).join("");
    $("#cmpNote").textContent=cmp.note;
  }catch(e){ /* comparisons appear after the first pipeline run */ }
}
main().catch(e=>{ $("#banner").innerHTML=`<b>Could not load the forecast.</b> ${esc(e.message)}`; });
</script>
</body></html>
"""
