"""
Stairway to Heaven - HTML-Report

Erzeugt nach jedem Lauf eine report.html im Ergebnisordner:
  - Kennzahlen (Pläne, Treppen, needs_review, Fehler)
  - Karten mit Vorschaubild pro Plan -> Klick öffnet Detailansicht mit großem Overlay
  - Tabelle aller Treppen mit Suche, Filtern und Sortierung
Die Seite ist komplett eigenständig: keine Internetverbindung, keine externen Dateien
außer den Ergebnisbildern im selben Ordner (relative Pfade).
"""

import html
import json
from datetime import datetime
from urllib.parse import quote


def _rel(path, run_dir):
    """Relativer, URL-kodierter Pfad (Leerzeichen, Umlaute, Sonderzeichen)."""
    if not path:
        return None
    rel = path.relative_to(run_dir).as_posix()
    return "/".join(quote(part) for part in rel.split("/"))


def write_report(run_dir, run_label, entries, settings, program_version, detector_version):
    """
    entries: Liste von dicts mit plan_name, status ('ok'/'error'), error, result (dict oder None),
             overlay (Path oder None), thumb (Path oder None)
    """
    plans = []
    for e in entries:
        r = e.get("result") or {}
        plans.append({
            "name": e["plan_name"],
            "status": e["status"],
            "error": e.get("error") or "",
            "overlay": _rel(e.get("overlay"), run_dir),
            "thumb": _rel(e.get("thumb"), run_dir) or _rel(e.get("overlay"), run_dir),
            "pages": r.get("pages", 1),
            "size": r.get("image_size", [0, 0]),
            "detections": r.get("detections", []),
        })
    data = {
        "run": run_label,
        "created": f"{datetime.now():%Y-%m-%d %H:%M}",
        "version": program_version,
        "detector": detector_version,
        "threshold": settings.get("confidence_threshold", 0.7),
        "yolo_threshold": (settings.get("yolo") or {}).get("review_threshold", 0.5),
        "method": settings.get("method", "rules"),
        "plans": plans,
    }
    page = TEMPLATE.replace("__TITLE__", html.escape(f"Stairway to Heaven – {run_label}"))
    page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    path = run_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f4f6fa; --card: #ffffff; --text: #16202e; --muted: #5d6b7e; --line: #dde3ec;
  --accent: #c8102e; --accent-soft: #fbe7ea; --ok: #1f8a4c; --ok-soft: #e3f4ea;
  --warn: #c86400; --warn-soft: #fff1e0; --err: #b3261e; --err-soft: #fde8e6;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 2px 8px rgba(16,24,40,.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #11151c; --card: #1a202a; --text: #e7ecf3; --muted: #97a3b5; --line: #2c3442;
    --accent: #ff5a6e; --accent-soft: #3a1d24; --ok: #4cc38a; --ok-soft: #173326;
    --warn: #ffa94d; --warn-soft: #3a2a14; --err: #ff7b72; --err-soft: #3b1d1b;
    --shadow: 0 1px 2px rgba(0,0,0,.4);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif; }
header { background: var(--card); border-bottom: 1px solid var(--line); }
.wrap { max-width: 1280px; margin: 0 auto; padding: 0 20px; }
.head { display: flex; align-items: center; gap: 14px; padding: 18px 0; flex-wrap: wrap; }
.logo { width: 38px; height: 38px; border-radius: 10px; background: var(--accent);
  display: grid; place-items: center; }
.logo svg { width: 24px; height: 24px; }
h1 { font-size: 20px; margin: 0; }
.sub { color: var(--muted); font-size: 13px; }
.head .meta { margin-left: auto; text-align: right; color: var(--muted); font-size: 13px; }
main { padding: 22px 0 60px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px; box-shadow: var(--shadow); }
.kpi b { display: block; font-size: 28px; line-height: 1.2; }
.kpi span { color: var(--muted); font-size: 13px; }
.kpi.warn b { color: var(--warn); } .kpi.err b { color: var(--err); } .kpi.acc b { color: var(--accent); }
h2 { font-size: 17px; margin: 30px 0 12px; display: flex; align-items: center; gap: 10px; }
h2 small { color: var(--muted); font-weight: 400; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
  box-shadow: var(--shadow); cursor: pointer; transition: transform .12s, border-color .12s;
  display: flex; flex-direction: column; }
.card:hover, .card:focus-visible { transform: translateY(-2px); border-color: var(--accent); outline: none; }
.thumb { height: 170px; background: #fff; display: grid; place-items: center; overflow: hidden;
  border-bottom: 1px solid var(--line); }
.thumb img { max-width: 100%; max-height: 100%; object-fit: contain; }
.thumb .noimg { color: var(--muted); font-size: 13px; }
.card .body { padding: 12px 14px; }
.card .name { font-weight: 600; word-break: break-word; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--bg);
  border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
.chip.ok { background: var(--ok-soft); color: var(--ok); border-color: transparent; }
.chip.warn { background: var(--warn-soft); color: var(--warn); border-color: transparent; }
.chip.err { background: var(--err-soft); color: var(--err); border-color: transparent; }
.chip.acc { background: var(--accent-soft); color: var(--accent); border-color: transparent; }
.toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
input[type=search], select { font: inherit; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--line);
  background: var(--card); color: var(--text); min-width: 0; }
input[type=search] { flex: 1 1 220px; }
label.check { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 14px; }
.tablebox { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: auto;
  box-shadow: var(--shadow); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--card); color: var(--muted); font-weight: 600;
  cursor: pointer; user-select: none; }
th:hover { color: var(--text); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:hover { background: var(--bg); }
tr.link { cursor: pointer; }
.empty { padding: 24px; text-align: center; color: var(--muted); }
/* detail view */
dialog { border: none; border-radius: 14px; padding: 0; width: min(1200px, 96vw); max-height: 94vh;
  background: var(--card); color: var(--text); box-shadow: 0 20px 60px rgba(0,0,0,.35); }
dialog::backdrop { background: rgba(10,14,20,.6); }
.dhead { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--line);
  position: sticky; top: 0; background: var(--card); z-index: 1; }
.dhead h3 { margin: 0; font-size: 17px; word-break: break-word; }
.dhead .sp { flex: 1; }
button, .btn { font: inherit; border: 1px solid var(--line); background: var(--card); color: var(--text);
  padding: 6px 12px; border-radius: 8px; cursor: pointer; text-decoration: none; font-size: 14px; }
button:hover, .btn:hover { border-color: var(--accent); color: var(--accent); }
.dbody { padding: 16px 18px; overflow: auto; max-height: calc(94vh - 64px); }
.viewer { background: #fff; border: 1px solid var(--line); border-radius: 10px; overflow: auto;
  max-height: 60vh; text-align: center; }
.viewer img { max-width: 100%; cursor: zoom-in; }
.viewer.zoom img { max-width: none; cursor: zoom-out; }
.hint { color: var(--muted); font-size: 12px; margin: 6px 0 14px; }
.legend { display: flex; gap: 14px; font-size: 13px; color: var(--muted); margin: 8px 0 0; }
.legend i { display: inline-block; width: 12px; height: 12px; border: 2px solid; margin-right: 5px;
  vertical-align: -1px; }
footer { color: var(--muted); font-size: 12px; text-align: center; padding: 20px; }
@media (max-width: 600px) { .head .meta { text-align: left; margin-left: 0; } }
</style>
</head>
<body>
<header><div class="wrap head">
  <div class="logo" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2"
    stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h5v-5h5v-5h5V5h3"/></svg></div>
  <div><h1>Stairway to Heaven</h1><div class="sub">Staircase detection in BVG floor plans · Result report</div></div>
  <div class="meta" id="meta"></div>
</div></header>

<main class="wrap">
  <section class="kpis" id="kpis"></section>

  <h2>Plans <small id="plancount"></small></h2>
  <div class="toolbar">
    <input type="search" id="planSearch" placeholder="Search plan …" aria-label="Search plan">
    <label class="check"><input type="checkbox" id="onlyReviewPlans"> only plans with needs_review</label>
    <label class="check"><input type="checkbox" id="onlyErrors"> only errors</label>
  </div>
  <div class="grid" id="cards"></div>

  <h2>All detected staircases <small id="detcount"></small></h2>
  <div class="toolbar">
    <input type="search" id="detSearch" placeholder="Search (plan, type …)" aria-label="Search staircases">
    <select id="viewFilter" aria-label="View">
      <option value="">all views</option><option value="stair_plan">Top view (stair_plan)</option>
      <option value="stair_section">Side view (stair_section)</option></select>
    <select id="typeFilter" aria-label="Stair type">
      <option value="">all stair types</option><option>straight</option><option>L-shaped</option>
      <option>U-shaped</option><option>other</option><option>unknown</option></select>
    <select id="methodFilter" aria-label="Method">
      <option value="">all methods</option><option value="rules">Rules</option>
      <option value="yolo">YOLO</option></select>
    <label class="check"><input type="checkbox" id="onlyReview"> only needs_review</label>
  </div>
  <div class="tablebox"><table>
    <thead><tr>
      <th data-k="plan">Plan</th><th data-k="id" class="num">ID</th><th data-k="view">View</th>
      <th data-k="stair_type">Type</th><th data-k="confidence" class="num">Confidence</th>
      <th data-k="needs_review">Review</th><th data-k="n_treads" class="num">Treads</th>
      <th data-k="page" class="num">Page</th><th data-k="method">Method</th>
      <th data-k="bbox">Box [x, y, w, h]</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table></div>
</main>

<dialog id="detail">
  <div class="dhead"><h3 id="dTitle"></h3><span class="sp"></span>
    <a class="btn" id="dOpen" target="_blank" rel="noopener">Open image in new tab</a>
    <button id="dClose" aria-label="Close">Close ✕</button></div>
  <div class="dbody">
    <div class="viewer" id="viewer"><img id="dImg" alt="Overlay"></div>
    <div class="legend"><span><i style="border-color:#e0182d"></i>Staircase (rules)</span>
      <span><i style="border-color:#ff8c00"></i>Rules: needs_review</span>
      <span class="ylegend"><i style="border-color:#005adc"></i>Staircase (YOLO)</span>
      <span class="ylegend"><i style="border-color:#00c8e6"></i>YOLO: needs_review</span></div>
    <div class="hint">Click the image to zoom (100 %). Box numbers in the image match the ID in the table.</div>
    <div class="tablebox"><table><thead><tr><th>ID</th><th>View</th><th>Type</th>
      <th class="num">Confidence</th><th>Review</th><th class="num">Treads</th><th class="num">Page</th>
      <th>Method</th><th>Box [x, y, w, h]</th></tr></thead><tbody id="dRows"></tbody></table></div>
    <p class="hint" id="dInfo"></p>
  </div>
</dialog>

<footer id="foot"></footer>

<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (x) => Number(x).toFixed(2);
const viewName = { stair_plan: "Top view", stair_section: "Side view" };
const methodName = { rules: "Rules", yolo: "YOLO", both: "Rules + YOLO" };

const plans = DATA.plans;
const allDets = plans.flatMap((p, pi) => p.detections.map(d => ({ ...d, method: d.method || "rules", plan: p.name, pi })));
const methodsUsed = [...new Set(allDets.map(d => d.method))];
const multi = DATA.method === "both";
const ok = plans.filter(p => p.status === "ok").length;
const errors = plans.length - ok;
const review = allDets.filter(d => d.needs_review).length;
const staircases = plans.reduce((n, p) => n + new Set(p.detections.map(d => d.staircase_id)).size, 0);

$("meta").innerHTML = `${esc(DATA.run)}<br>${esc(DATA.created)} · Program ${esc(DATA.version)} · Detector ${esc(DATA.detector)} · Method: ${esc(methodName[DATA.method] || DATA.method)}`;
$("foot").textContent = (DATA.method !== "yolo" ? `Rules: threshold ${fmt(DATA.threshold)} ("confidence" is a rule score, not a probability) · ` : "") +
  (DATA.method !== "rules" ? `YOLO: threshold ${fmt(DATA.yolo_threshold)} (model confidence) · ` : "") + "Report runs locally without internet";
const kpis = [
  ["Plans processed", `${ok} / ${plans.length}`, ""],
  ["Staircase flights (boxes)", allDets.length, "acc"],
];
if (multi) {
  kpis.push(["of which rules", allDets.filter(d => d.method === "rules").length, ""]);
  kpis.push(["of which YOLO", allDets.filter(d => d.method === "yolo").length, ""]);
} else {
  kpis.push(["Staircases (grouped)", staircases, ""]);
}
kpis.push(["needs_review", review, review ? "warn" : ""], ["Errors", errors, errors ? "err" : ""]);
$("kpis").innerHTML = kpis.map(([l, v, c]) => `<div class="kpi ${c}"><b>${v}</b><span>${l}</span></div>`).join("");
if (methodsUsed.length < 2 && !multi) $("methodFilter").style.display = "none";
document.querySelectorAll(".ylegend").forEach(e => e.style.display = DATA.method === "rules" ? "none" : "");

function renderCards() {
  const q = $("planSearch").value.toLowerCase();
  const onlyRev = $("onlyReviewPlans").checked, onlyErr = $("onlyErrors").checked;
  const list = plans.map((p, i) => ({ p, i })).filter(({ p }) =>
    p.name.toLowerCase().includes(q) &&
    (!onlyRev || p.detections.some(d => d.needs_review)) &&
    (!onlyErr || p.status !== "ok"));
  $("plancount").textContent = `${list.length} of ${plans.length}`;
  $("cards").innerHTML = list.map(({ p, i }) => {
    const n = p.detections.length, r = p.detections.filter(d => d.needs_review).length;
    const sec = p.detections.filter(d => d.view === "stair_section").length;
    const nr = p.detections.filter(d => (d.method || "rules") === "rules").length, ny = n - nr;
    const chips = p.status !== "ok"
      ? `<span class="chip err">Error</span>`
      : (multi ? `<span class="chip acc">Rules ${nr}</span><span class="chip acc">YOLO ${ny}</span>`
               : `<span class="chip ${n ? "acc" : ""}">${n} staircase flight${n === 1 ? "" : "s"}</span>`) +
        (sec ? `<span class="chip">${sec} side view${sec === 1 ? "" : "s"}</span>` : "") +
        (r ? `<span class="chip warn">${r} to review</span>` : `<span class="chip ok">ok</span>`) +
        (p.pages > 1 ? `<span class="chip">${p.pages} pages</span>` : "");
    const img = p.thumb ? `<img loading="lazy" src="${p.thumb}" alt="Preview ${esc(p.name)}">`
                        : `<span class="noimg">no image</span>`;
    return `<div class="card" tabindex="0" data-i="${i}"><div class="thumb">${img}</div>
      <div class="body"><div class="name">${esc(p.name)}</div><div class="chips">${chips}</div></div></div>`;
  }).join("") || `<div class="empty">No plans found.</div>`;
}

let sortKey = "plan", sortDir = 1;
function detRow(d, withPlan) {
  const box = `[${d.bbox.join(", ")}]`;
  const rev = d.needs_review ? `<span class="chip warn">yes</span>` : `<span class="chip ok">no</span>`;
  return `<tr ${withPlan ? `class="link" data-i="${d.pi}"` : ""}>` +
    (withPlan ? `<td>${esc(d.plan)}</td>` : "") +
    `<td class="num">${d.id}</td><td>${viewName[d.view] || esc(d.view)}</td>` +
    `<td>${esc(d.stair_type || "–")}</td><td class="num">${fmt(d.confidence)}</td><td>${rev}</td>` +
    `<td class="num">${d.n_treads ?? ""}</td><td class="num">${d.page ?? 1}</td>` +
    `<td>${esc(methodName[d.method || "rules"] || d.method)}</td><td>${box}</td></tr>`;
}
function renderRows() {
  const q = $("detSearch").value.toLowerCase(), v = $("viewFilter").value, t = $("typeFilter").value;
  const m = $("methodFilter").value, onlyRev = $("onlyReview").checked;
  let list = allDets.filter(d =>
    (!q || `${d.plan} ${d.view} ${d.stair_type} ${methodName[d.method]}`.toLowerCase().includes(q)) &&
    (!v || d.view === v) && (!t || d.stair_type === t) && (!m || d.method === m) && (!onlyRev || d.needs_review));
  list.sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (typeof x === "number" && typeof y === "number") return (x - y) * sortDir;
    return String(x ?? "").localeCompare(String(y ?? ""), "en", { numeric: true }) * sortDir;
  });
  $("detcount").textContent = `${list.length} of ${allDets.length}`;
  $("rows").innerHTML = list.map(d => detRow(d, true)).join("") ||
    `<tr><td colspan="10" class="empty">No staircases for this selection.</td></tr>`;
}

function openDetail(i) {
  const p = plans[i];
  $("dTitle").textContent = p.name;
  $("viewer").classList.remove("zoom");
  const has = !!p.overlay;
  $("viewer").style.display = has ? "" : "none";
  document.querySelector(".legend").style.display = has ? "" : "none";
  document.querySelector(".dbody .hint").style.display = has ? "" : "none";
  $("dOpen").style.display = has ? "" : "none";
  if (has) { $("dImg").src = p.overlay; $("dOpen").href = p.overlay; } else { $("dImg").removeAttribute("src"); }
  $("dRows").innerHTML = p.detections.map(d => detRow({ ...d }, false)).join("") ||
    `<tr><td colspan="9" class="empty">${p.status === "ok" ? "No staircase detected." : "The plan could not be processed."}</td></tr>`;
  $("dInfo").textContent = p.status === "ok"
    ? `Original size: ${p.size[0]} × ${p.size[1]} pixels · coordinates refer to the original TIF.`
    : `Error: ${p.error}`;
  $("detail").showModal();
}

$("cards").addEventListener("click", e => { const c = e.target.closest(".card"); if (c) openDetail(+c.dataset.i); });
$("cards").addEventListener("keydown", e => { if (e.key === "Enter") { const c = e.target.closest(".card"); if (c) openDetail(+c.dataset.i); } });
$("rows").addEventListener("click", e => { const r = e.target.closest("tr.link"); if (r) openDetail(+r.dataset.i); });
$("dClose").onclick = () => $("detail").close();
$("detail").addEventListener("click", e => { if (e.target === $("detail")) $("detail").close(); });
$("dImg").onclick = () => $("viewer").classList.toggle("zoom");
document.querySelectorAll("thead th[data-k]").forEach(th => th.onclick = () => {
  sortDir = sortKey === th.dataset.k ? -sortDir : 1; sortKey = th.dataset.k; renderRows();
});
["planSearch", "onlyReviewPlans", "onlyErrors"].forEach(id => $(id).addEventListener("input", renderCards));
["detSearch", "viewFilter", "typeFilter", "methodFilter", "onlyReview"].forEach(id => $(id).addEventListener("input", renderRows));
renderCards(); renderRows();
</script>
</body>
</html>
"""
