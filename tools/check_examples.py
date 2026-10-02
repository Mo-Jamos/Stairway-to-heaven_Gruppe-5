"""
Prüft alle Beispielbilder in data/examples/: Welche Treppen erkennen die Regeln, welche YOLO?

    python tools/check_examples.py

Ergebnis: output/<Datum_Uhrzeit>_beispiele/
    beispiele.html   Bericht im Browser (Bild mit Boxen, erkannt ja/nein, Typ, Konfidenz)
    beispiele.csv    dieselben Angaben als Tabelle (Excel)
    *_check.png      Beispielbild mit Boxen (rot = Regeln, blau = YOLO)

So seht ihr sofort, welche Treppenarten noch NICHT erkannt werden -> genau diese Beispiele
helfen beim YOLO-Training am meisten (bzw. zeigen, welche Regel fehlt).
Bei keine_treppe/ ist "nichts erkannt" das richtige Ergebnis.
"""
import csv
import html
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import examples as ex          # noqa: E402
import stairway_app as core    # noqa: E402
import yolo_detector           # noqa: E402

RED, ORANGE, BLUE = (0, 0, 230), (0, 140, 255), (220, 90, 0)
TYPE_CHECKABLE = {"straight", "L-shaped", "U-shaped"}


def yolo_best(gray, folder, settings, rules_scale):
    """YOLO in der Größe der Regeln (oder mehreren Größen). Rückgabe: (Größe, Boxen in Beispiel-Pixeln)."""
    view = ex.FOLDERS[folder][0]
    scales = [rules_scale] if rules_scale != 1.0 else [s for s in (0.75, 1.0, 1.5, 2.0)
                                                        if s in ex.usable_scales(gray)]
    best = (1.0, [], -1.0)
    for s in scales:
        boxes = list(yolo_detector.detect_gray(ex.resize(gray, s), settings))
        for b in boxes:
            b["bbox"] = [v / s for v in b["bbox"]]
        wanted = [b["confidence"] for b in boxes if view is None or b["view"] == view]
        top = max(wanted) if wanted else (0.0 if view is None and not boxes else -0.5)
        if top > best[2]:
            best = (s, boxes, top)
    return best[0], best[1]


def draw(gray, rules, yolo, path):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    f = 1.0 if max(img.shape) >= 300 else 300 / max(img.shape)
    if f != 1.0:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    for t in rules:
        x1, y1, x2, y2 = [int(round(v * f)) for v in t["bbox_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
    for b in yolo:
        x1, y1, x2, y2 = [int(round(v * f)) for v in b["bbox"]]
        cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), BLUE, 2)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        path.write_bytes(buf.tobytes())


def main():
    settings = core.load_settings()
    ex.create_folders()
    items = ex.list_examples()
    if not items:
        print(f"Keine Beispielbilder gefunden. Bilder in die Unterordner von {ex.EXAMPLES_DIR} legen.")
        return
    yolo_ok, yolo_msg = yolo_detector.status(settings)
    print(f"{len(items)} Beispiele · YOLO: {'ja' if yolo_ok else 'nein (' + yolo_msg.splitlines()[0] + ')'}")
    run_dir = core.new_run_folder("beispiele")
    rows = []
    for i, (folder, path) in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {folder}/{path.name}")
        view, typ = ex.FOLDERS[folder]
        row = {"folder": folder, "file": path.name, "expected_view": view or "keine Treppe",
               "expected_type": typ or ""}
        try:
            gray = ex.load_gray(path)
            res = ex.rules_best(gray, folder)
            best = res["best"]
            if view is None:
                hits = {s: n for s, n in res["any_scale_hits"].items() if n}
                row["rules"] = "ok (nichts erkannt)" if not hits else \
                    "FEHLALARM bei Größe " + ", ".join(f"x{s}" for s in hits)
                row["rules_ok"] = not hits
            else:
                row["rules_ok"] = best is not None
                row["rules"] = "erkannt" if best else "NICHT erkannt"
            row["rules_conf"] = best["confidence"] if best else ""
            row["rules_type"] = (best.get("stair_type") or "") if best else ""
            row["type_ok"] = ("ja" if row["rules_type"] == typ else "nein") \
                if best and typ in TYPE_CHECKABLE else "–"
            row["scale"] = res["scale"]
            rules_draw = res["stairs"] if view is not None else []
            yolo_boxes = []
            if yolo_ok:
                _, yolo_boxes = yolo_best(gray, folder, settings, res["scale"])
                wanted = [b for b in yolo_boxes if view is None or b["view"] == view]
                if view is None:
                    row["yolo_ok"] = not yolo_boxes
                    row["yolo"] = "ok (nichts erkannt)" if not yolo_boxes else "FEHLALARM"
                else:
                    row["yolo_ok"] = bool(wanted)
                    row["yolo"] = "erkannt" if wanted else "NICHT erkannt"
                row["yolo_conf"] = round(max(b["confidence"] for b in wanted), 2) if wanted else ""
            else:
                row["yolo"], row["yolo_ok"], row["yolo_conf"] = "–", None, ""
            img_name = f"{core.safe_name(folder)}__{core.safe_name(path.stem)}_check.png"
            draw(gray, rules_draw, yolo_boxes, run_dir / img_name)
            row["image"] = img_name
        except Exception as e:                                    # noqa: BLE001
            row.update(rules=f"FEHLER: {e}", rules_ok=False, yolo="–", yolo_ok=None, image=None,
                       rules_conf="", rules_type="", type_ok="–", scale="", yolo_conf="")
        rows.append(row)

    write_csv(run_dir, rows, settings)
    report = write_html(run_dir, rows, yolo_ok)
    print("\nZusammenfassung (Regeln):")
    for folder in ex.FOLDERS:
        part = [r for r in rows if r["folder"] == folder]
        if part:
            good = sum(bool(r["rules_ok"]) for r in part)
            yolo = sum(bool(r["yolo_ok"]) for r in part) if yolo_ok else None
            print(f"  {folder:<14} {good}/{len(part)} richtig" + (f" · YOLO {yolo}/{len(part)}" if yolo_ok else ""))
    print(f"\nBericht: {report}")


def write_csv(run_dir, rows, settings):
    delim, dec = (";", ",") if settings.get("csv_locale", "de") == "de" else (",", ".")
    keys = ["folder", "file", "expected_view", "expected_type", "rules", "rules_conf", "rules_type",
            "type_ok", "scale", "yolo", "yolo_conf"]
    with open(run_dir / "beispiele.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=delim)
        w.writerow(keys)
        for r in rows:
            w.writerow([str(r.get(k, "")).replace(".", dec) if isinstance(r.get(k), float) else r.get(k, "")
                        for k in keys])


def write_html(run_dir, rows, yolo_ok):
    def badge(ok, text):
        cls = "na" if ok is None else ("ok" if ok else "bad")
        return f'<span class="b {cls}">{html.escape(str(text))}</span>'

    summary = []
    for folder in ex.FOLDERS:
        part = [r for r in rows if r["folder"] == folder]
        if part:
            y = f"{sum(bool(r['yolo_ok']) for r in part)}/{len(part)}" if yolo_ok else "–"
            summary.append(f"<tr><td>{folder}</td><td>{len(part)}</td>"
                           f"<td>{sum(bool(r['rules_ok']) for r in part)}/{len(part)}</td><td>{y}</td></tr>")
    cards = []
    for r in rows:
        img = (f'<a href="{quote(r["image"])}" target="_blank"><img src="{quote(r["image"])}" alt=""></a>'
               if r.get("image") else '<div class="noimg">kein Bild</div>')
        conf = f' · {r["rules_conf"]:.2f}'.replace(".", ",") if isinstance(r.get("rules_conf"), float) else ""
        typ = f'<div class="m">Typ (Regeln): {html.escape(r["rules_type"] or "–")} · richtig: {r["type_ok"]}</div>' \
            if r["expected_view"] == "stair_plan" else ""
        ytxt = f'{r["yolo"]}' + (f' · {r["yolo_conf"]}'.replace(".", ",") if r.get("yolo_conf") != "" else "")
        cards.append(f'''<div class="card"><div class="img">{img}</div><div class="body">
<div class="n">{html.escape(r["folder"])}/{html.escape(r["file"])}</div>
<div class="m">Regeln: {badge(r["rules_ok"], r["rules"] + conf)}</div>{typ}
<div class="m">YOLO: {badge(r["yolo_ok"], ytxt)}</div>
<div class="m small">geprüfte Größe: x{r.get("scale", "")}</div></div></div>''')
    page = f'''<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stairway to Heaven – Beispiele</title>
<style>
:root{{--bg:#f4f6fa;--card:#fff;--t:#16202e;--mu:#5d6b7e;--li:#dde3ec}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11151c;--card:#1a202a;--t:#e7ecf3;--mu:#97a3b5;--li:#2c3442}}}}
body{{margin:0;background:var(--bg);color:var(--t);font:15px/1.5 "Segoe UI",system-ui,Arial,sans-serif}}
.w{{max-width:1200px;margin:0 auto;padding:20px}} h1{{font-size:21px;margin:0 0 4px}} .mu{{color:var(--mu)}}
table{{border-collapse:collapse;background:var(--card);border:1px solid var(--li);border-radius:10px;margin:14px 0}}
td,th{{padding:6px 14px;border-bottom:1px solid var(--li);text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--li);border-radius:12px;overflow:hidden}}
.img{{background:#fff;height:190px;display:grid;place-items:center;border-bottom:1px solid var(--li)}}
.img img{{max-width:100%;max-height:190px}} .body{{padding:10px 12px}} .n{{font-weight:600;word-break:break-all}}
.m{{margin-top:4px;font-size:14px}} .small{{font-size:12px;color:var(--mu)}}
.b{{padding:1px 8px;border-radius:999px;font-size:13px}} .ok{{background:#e3f4ea;color:#1f8a4c}}
.bad{{background:#fde8e6;color:#b3261e}} .na{{background:var(--bg);color:var(--mu)}}
</style></head><body><div class="w">
<h1>Beispiele prüfen</h1><div class="mu">{datetime.now():%d.%m.%Y %H:%M} · rote Box = Regeln, blaue Box = YOLO ·
Bild anklicken = groß</div>
<table><tr><th>Ordner</th><th>Bilder</th><th>Regeln richtig</th><th>YOLO richtig</th></tr>{"".join(summary)}</table>
<div class="grid">{"".join(cards)}</div></div></body></html>'''
    path = run_dir / "beispiele.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
