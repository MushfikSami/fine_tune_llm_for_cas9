#!/usr/bin/env python
"""Step 5b: render an interactive HTML report from designed_sgrnas.csv.

Produces a single self-contained HTML file (no external CDNs / internet needed):
  - headline stat tiles,
  - an interactive scatter of on-target efficiency vs off-target count
    (hover tooltips, coloured by strand),
  - an on-target efficiency histogram,
  - a sortable / filterable guide table with per-guide safety tiers.

Usage:
    python 05_visualize.py                       # reads designed_sgrnas.csv
    python 05_visualize.py --csv path.csv --out report.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sgRNA Design Report — __GENE__</title>
<style>
  :root {
    --surface-0: #f4f4f2; --surface-1: #fcfcfb; --surface-2: #efeeeb;
    --border: #e2e1dc; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --text-muted: #86857f;
    --series-fwd: #2a78d6; --series-rev: #1baf7a;
    --seq-100:#cde2fb; --seq-400:#3987e5; --seq-600:#184f95;
    --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
    --shadow: 0 1px 2px rgba(0,0,0,.06), 0 2px 8px rgba(0,0,0,.04);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-0:#141413; --surface-1:#1a1a19; --surface-2:#232322;
      --border:#33322f; --text-primary:#ffffff; --text-secondary:#c3c2b7;
      --text-muted:#8a8980;
      --series-fwd:#3987e5; --series-rev:#199e70;
      --seq-100:#184f95; --seq-400:#3987e5; --seq-600:#9ec5f4;
      --shadow: 0 1px 2px rgba(0,0,0,.4), 0 2px 10px rgba(0,0,0,.3);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin:0; background:var(--surface-0); color:var(--text-primary);
    font: 14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:1180px; margin:0 auto; padding:28px 22px 60px; }
  header h1 { margin:0 0 4px; font-size:22px; letter-spacing:-.01em; }
  header p { margin:0; color:var(--text-secondary); }
  .mono { font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace; }

  .tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:24px 0; }
  .tile { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
          padding:16px 18px; box-shadow:var(--shadow); }
  .tile .k { color:var(--text-secondary); font-size:12px; text-transform:uppercase;
             letter-spacing:.04em; }
  .tile .v { font-size:26px; font-weight:650; margin-top:6px; letter-spacing:-.02em; }
  .tile .s { color:var(--text-muted); font-size:12px; margin-top:2px; }

  .panels { display:grid; grid-template-columns:1.5fr 1fr; gap:16px; margin-bottom:24px; }
  @media (max-width:820px){ .panels{grid-template-columns:1fr;} .tiles{grid-template-columns:repeat(2,1fr);} }
  .card { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
          padding:16px 18px; box-shadow:var(--shadow); }
  .card h2 { margin:0 0 2px; font-size:14px; }
  .card .sub { color:var(--text-muted); font-size:12px; margin-bottom:8px; }
  svg { display:block; width:100%; height:auto; overflow:visible; }
  .axis line, .axis path { stroke:var(--border); stroke-width:1; }
  .axis text { fill:var(--text-muted); font-size:11px; }
  .gridline { stroke:var(--border); stroke-width:1; opacity:.55; }
  .dot { stroke:var(--surface-1); stroke-width:1.5; cursor:pointer; }
  .legend { display:flex; gap:16px; align-items:center; margin-top:10px; font-size:12px;
            color:var(--text-secondary); }
  .legend .sw { display:inline-block; width:10px; height:10px; border-radius:50%;
                margin-right:6px; vertical-align:middle; }

  .toolbar { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center;
             margin-bottom:12px; }
  .toolbar label { font-size:12px; color:var(--text-secondary); display:flex;
                   align-items:center; gap:6px; }
  input[type=search], select { background:var(--surface-1); color:var(--text-primary);
     border:1px solid var(--border); border-radius:8px; padding:6px 9px; font-size:13px; }
  input[type=search]{ min-width:200px; }
  input[type=range]{ accent-color:var(--seq-400); }

  table { width:100%; border-collapse:collapse; font-size:13px; }
  thead th { position:sticky; top:0; background:var(--surface-2); text-align:left;
     padding:9px 10px; border-bottom:1px solid var(--border); cursor:pointer;
     white-space:nowrap; user-select:none; font-weight:600; }
  thead th .arrow { color:var(--text-muted); font-size:10px; }
  tbody td { padding:8px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
  tbody tr:hover { background:var(--surface-2); }
  .rank { color:var(--text-muted); }
  .bar { display:inline-block; height:8px; border-radius:4px; background:var(--seq-400);
         vertical-align:middle; margin-right:8px; }
  .badge { display:inline-flex; align-items:center; gap:5px; padding:2px 8px;
           border-radius:999px; font-size:11px; font-weight:600; border:1px solid transparent; }
  .b-good{ color:var(--good); border-color:var(--good); }
  .b-warning{ color:var(--warning); border-color:var(--warning); }
  .b-serious{ color:var(--serious); border-color:var(--serious); }
  .b-critical{ color:var(--critical); border-color:var(--critical); }

  #tt { position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
        background:var(--surface-1); border:1px solid var(--border); border-radius:9px;
        box-shadow:var(--shadow); padding:9px 11px; font-size:12px; z-index:9; max-width:240px; }
  #tt b { font-family:"SF Mono",ui-monospace,monospace; }
  .foot { color:var(--text-muted); font-size:12px; margin-top:26px; }
</style>
</head>
<body data-palette="__PALETTE__">
<div class="wrap">
  <header>
    <h1>sgRNA Design Report</h1>
    <p>Target gene <span class="mono">__GENE__</span> · SpCas9 (NGG PAM) ·
       on-target by fine-tuned DNABERT-2, off-target by bowtie2</p>
  </header>

  <section class="tiles" id="tiles"></section>

  <section class="panels">
    <div class="card">
      <h2>Efficiency vs. off-target risk</h2>
      <div class="sub">Best guides sit lower-right: high predicted efficiency, few off-targets. Hover a point.</div>
      <div id="scatter"></div>
      <div class="legend">
        <span><span class="sw" style="background:var(--series-fwd)"></span>Forward (+) strand</span>
        <span><span class="sw" style="background:var(--series-rev)"></span>Reverse (−) strand</span>
      </div>
    </div>
    <div class="card">
      <h2>On-target score distribution</h2>
      <div class="sub">Predicted efficiency across all candidates.</div>
      <div id="hist"></div>
    </div>
  </section>

  <div class="card">
    <div class="toolbar">
      <input type="search" id="q" placeholder="Search spacer / PAM…">
      <label>Strand
        <select id="fStrand"><option value="">all</option><option value="+">+ only</option><option value="-">− only</option></select>
      </label>
      <label>Max off-targets <span id="offVal" class="mono"></span>
        <input type="range" id="fOff" min="0" step="1">
      </label>
      <label><input type="checkbox" id="fClean"> zero off-target only</label>
      <span id="count" style="margin-left:auto; color:var(--text-muted); font-size:12px;"></span>
    </div>
    <div style="overflow:auto; max-height:560px;">
      <table id="tbl">
        <thead><tr id="head"></tr></thead>
        <tbody id="body"></tbody>
      </table>
    </div>
  </div>

  <p class="foot">Scores are a transferable-biochemistry prior (Doench 2016), not rice-validated absolutes — use them to rank. Validate top guides experimentally.</p>
</div>
<div id="tt"></div>

<script>
const DATA = __DATA__;
const GENE = "__GENE__";

// ---- helpers ----
const fmt = (x,d=3)=> (x==null||isNaN(x))? "–" : Number(x).toFixed(d);
const maxOff = Math.max(1, ...DATA.map(d=>d.offtarget_count));
const onVals = DATA.map(d=>d.on_target);
const onMin = Math.min(...onVals), onMax = Math.max(...onVals);
function tier(off){
  if(off===0)  return ["good","● safe"];
  if(off<=3)   return ["warning","▲ low"];
  if(off<=9)   return ["serious","◆ moderate"];
  return ["critical","✖ high"];
}
const cssVar = n=>getComputedStyle(document.body).getPropertyValue(n).trim();

// ---- stat tiles ----
(function(){
  const clean = DATA.filter(d=>d.offtarget_count===0).length;
  const best = DATA.reduce((a,b)=> b.final_score>a.final_score? b:a, DATA[0]);
  const medGC = [...DATA].map(d=>d.gc_percent).sort((a,b)=>a-b)[Math.floor(DATA.length/2)];
  const tiles = [
    ["Candidate guides", DATA.length, "both strands, NGG PAM"],
    ["Zero off-target", clean, (100*clean/DATA.length).toFixed(0)+"% of candidates"],
    ["Top on-target", fmt(best.on_target), "spacer "+best.spacer.slice(0,8)+"…"],
    ["Median GC", medGC+"%", "duplex stability check"],
  ];
  document.getElementById("tiles").innerHTML = tiles.map(t=>
    `<div class="tile"><div class="k">${t[0]}</div><div class="v">${t[1]}</div><div class="s">${t[2]}</div></div>`).join("");
})();

// ---- tooltip ----
const tt = document.getElementById("tt");
function showTT(html,x,y){ tt.innerHTML=html; tt.style.opacity=1;
  tt.style.left=Math.min(x+14, innerWidth-250)+"px"; tt.style.top=(y+14)+"px"; }
function hideTT(){ tt.style.opacity=0; }

// ---- scatter (SVG, on_target x, offtarget y sqrt-ish) ----
function scatter(){
  const W=560,H=300,m={l:44,r:14,t:12,b:38};
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const x0=onMin-0.01, x1=onMax+0.01;
  const sx = v => m.l + (v-x0)/(x1-x0)*iw;
  const sy = v => m.t + ih - (Math.sqrt(v)/Math.sqrt(maxOff))*ih;   // sqrt spreads the 0-heavy axis
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Efficiency vs off-target scatter">`;
  // gridlines + y ticks
  const yticks=[0,1,4,9,Math.round(maxOff)].filter((v,i,a)=>a.indexOf(v)===i && v<=maxOff);
  yticks.forEach(v=>{ const y=sy(v);
    s+=`<line class="gridline" x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}"/>`+
       `<text class="mono" x="${m.l-8}" y="${y+3}" text-anchor="end" fill="${cssVar('--text-muted')}" font-size="11">${v}</text>`; });
  // x ticks
  for(let k=0;k<=4;k++){ const v=x0+(x1-x0)*k/4, x=sx(v);
    s+=`<line class="gridline" x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t+ih}"/>`+
       `<text x="${x}" y="${H-14}" text-anchor="middle" fill="${cssVar('--text-muted')}" font-size="11">${v.toFixed(2)}</text>`; }
  s+=`<text x="${m.l+iw/2}" y="${H-1}" text-anchor="middle" fill="${cssVar('--text-secondary')}" font-size="12">on-target efficiency →</text>`;
  s+=`<text transform="translate(13,${m.t+ih/2}) rotate(-90)" text-anchor="middle" fill="${cssVar('--text-secondary')}" font-size="12">off-target count</text>`;
  // points (reverse-sorted so best drawn last/on top)
  const pts=[...DATA].sort((a,b)=>a.final_score-b.final_score);
  pts.forEach(d=>{ const c = d.strand==="+"? cssVar('--series-fwd'):cssVar('--series-rev');
    s+=`<circle class="dot" cx="${sx(d.on_target).toFixed(1)}" cy="${sy(d.offtarget_count).toFixed(1)}" r="5" fill="${c}"`+
       ` data-i="${d.__i}"/>`; });
  s+=`</svg>`;
  document.getElementById("scatter").innerHTML=s;
  document.querySelectorAll("#scatter .dot").forEach(el=>{
    el.addEventListener("mousemove",e=>{ const d=DATA[+el.dataset.i]; const [cls,lab]=tier(d.offtarget_count);
      showTT(`<b>${d.spacer}</b><br>PAM ${d.pam} · ${d.strand} · pos ${d.position}<br>`+
             `on-target <b>${fmt(d.on_target)}</b> · off-targets <b>${d.offtarget_count}</b> (${lab.split(' ')[1]})<br>`+
             `final <b>${fmt(d.final_score)}</b>`, e.clientX,e.clientY); });
    el.addEventListener("mouseleave",hideTT);
  });
}

// ---- histogram of on_target ----
function hist(){
  const W=360,H=300,m={l:34,r:12,t:12,b:38}, bins=12;
  const iw=W-m.l-m.r, ih=H-m.t-m.b;
  const lo=onMin, hi=onMax+1e-9, bw=(hi-lo)/bins;
  const counts=new Array(bins).fill(0);
  DATA.forEach(d=>{ let b=Math.floor((d.on_target-lo)/bw); if(b<0)b=0; if(b>=bins)b=bins-1; counts[b]++; });
  const cmax=Math.max(...counts);
  const bx = i => m.l + i/bins*iw;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="On-target histogram">`;
  [0,0.5,1].forEach(f=>{ const v=Math.round(cmax*f), y=m.t+ih-(v/cmax)*ih;
    s+=`<line class="gridline" x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}"/>`+
       `<text class="mono" x="${m.l-6}" y="${y+3}" text-anchor="end" fill="${cssVar('--text-muted')}" font-size="11">${v}</text>`; });
  counts.forEach((c,i)=>{ const h=(c/cmax)*ih, x=bx(i)+2, w=iw/bins-3, y=m.t+ih-h;
    s+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(0,h).toFixed(1)}"`+
       ` rx="3" fill="${cssVar('--seq-400')}" data-b="${i}"/>`; });
  for(let k=0;k<=3;k++){ const v=lo+(hi-lo)*k/3, x=m.l+iw*k/3;
    s+=`<text x="${x}" y="${H-14}" text-anchor="middle" fill="${cssVar('--text-muted')}" font-size="11">${v.toFixed(2)}</text>`; }
  s+=`<text x="${m.l+iw/2}" y="${H-1}" text-anchor="middle" fill="${cssVar('--text-secondary')}" font-size="12">on-target efficiency</text>`;
  s+=`</svg>`;
  document.getElementById("hist").innerHTML=s;
  document.querySelectorAll("#hist rect").forEach(el=>{
    el.addEventListener("mousemove",e=>{ const i=+el.dataset.b;
      showTT(`${counts[i]} guides<br>${(lo+i*bw).toFixed(2)}–${(lo+(i+1)*bw).toFixed(2)}`,e.clientX,e.clientY); });
    el.addEventListener("mouseleave",hideTT);
  });
}

// ---- table ----
const COLS=[
  ["#","rank"],["spacer (5′→3′)","spacer"],["PAM","pam"],["str","strand"],
  ["pos","position"],["GC%","gc_percent"],["on-target","on_target"],
  ["off-tgt","offtarget_count"],["safety","tier"],["final","final_score"],
];
let sortKey="final_score", sortDir=-1;
function rowsFiltered(){
  const q=document.getElementById("q").value.trim().toUpperCase();
  const st=document.getElementById("fStrand").value;
  const mo=+document.getElementById("fOff").value;
  const clean=document.getElementById("fClean").checked;
  return DATA.filter(d=>
    (!q || d.spacer.includes(q) || d.pam.includes(q)) &&
    (!st || d.strand===st) &&
    (d.offtarget_count<=mo) &&
    (!clean || d.offtarget_count===0));
}
function renderHead(){
  document.getElementById("head").innerHTML = COLS.map(c=>{
    const arrow = c[1]===sortKey ? (sortDir<0?" ▼":" ▲") : "";
    return `<th data-k="${c[1]}">${c[0]}<span class="arrow">${arrow}</span></th>`;}).join("");
  document.querySelectorAll("#head th").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(k==="tier")return;
    if(sortKey===k) sortDir*=-1; else {sortKey=k; sortDir=(k==="spacer"||k==="pam")?1:-1;}
    renderHead(); renderBody();
  });
}
function renderBody(){
  let rows=rowsFiltered();
  rows.sort((a,b)=>{ let x=a[sortKey],y=b[sortKey];
    if(typeof x==="string") return sortDir*x.localeCompare(y);
    return sortDir*((x-y)); });
  const onSpan=(onMax-onMin)||1;
  document.getElementById("body").innerHTML = rows.map((d)=>{
    const [cls,lab]=tier(d.offtarget_count);
    const bw=6+38*(d.on_target-onMin)/onSpan;
    return `<tr>
      <td class="rank">${d.__rank}</td>
      <td class="mono">${d.spacer}</td>
      <td class="mono">${d.pam}</td>
      <td>${d.strand}</td>
      <td class="mono">${d.position}</td>
      <td>${d.gc_percent}</td>
      <td><span class="bar" style="width:${bw.toFixed(0)}px"></span>${fmt(d.on_target)}</td>
      <td>${d.offtarget_count}</td>
      <td><span class="badge b-${cls}">${lab}</span></td>
      <td class="mono">${fmt(d.final_score)}</td>
    </tr>`;}).join("");
  document.getElementById("count").textContent = rows.length+" of "+DATA.length+" guides";
}

// ---- init ----
DATA.forEach((d,i)=>{d.__i=i;});
[...DATA].sort((a,b)=>b.final_score-a.final_score).forEach((d,i)=>{d.__rank=i+1;});
const off=document.getElementById("fOff"); off.max=maxOff; off.value=maxOff;
document.getElementById("offVal").textContent=maxOff;
off.oninput=()=>{document.getElementById("offVal").textContent=off.value; renderBody();};
["q","fStrand","fClean"].forEach(id=>document.getElementById(id).oninput=renderBody);
scatter(); hist(); renderHead(); renderBody();
addEventListener("resize",()=>{scatter();hist();});
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="designed_sgrnas.csv")
    ap.add_argument("--out", default="designed_sgrnas.html")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(f"{csv} not found — run 04_design_sgrna.py first.")
    df = pd.read_csv(csv)
    gene = str(df["gene"].iloc[0]) if "gene" in df.columns and len(df) else "target"
    records = df.to_dict(orient="records")

    palette = "#2a78d6,#1baf7a"
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(records))
            .replace("__GENE__", gene)
            .replace("__PALETTE__", palette))
    Path(args.out).write_text(html)
    print(f"Wrote interactive report to {args.out} ({len(df)} guides, gene {gene})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
