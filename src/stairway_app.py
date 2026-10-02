"""
Stairway to Heaven - Programm-Logik (ohne Fenster)

Setzt "How the program works" um:
  - Ordner plans/ (Eingabe) und output/ (Ergebnisse) neben dem Programm
  - Einzelplan oder Batch, jeder Lauf in einem eigenen Unterordner mit Datum/Uhrzeit
  - pro Plan: <plan>_overlay.png und <plan>_result.json
  - pro Lauf: all_detections.csv, report.html (Ergebnisbericht im Browser),
    bei Batch zusätzlich summary_all_plans.csv und processing_log.txt
  - needs_review, wenn confidence < confidence_threshold (settings.json)
  - Fehler in einem Plan stoppen den Batch nicht
  - zwei Erkennungsmethoden (settings.json "method" oder Umschalter im Fenster):
      "rules" = Regeln (src/stair_detector.py), "yolo" = YOLO (src/yolo_detector.py),
      "both"  = beide gleichzeitig zum Vergleichen (Feld "method" pro Treppe)

Wird von app_modern.py (Fenster) und main.py (Terminal) benutzt.
"""

import csv
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2

import report
import stair_detector as sd
import yolo_detector

PROGRAM_VERSION = "0.4"
BASE_DIR = sd.PROJECT_DIR
PLANS_DIR = BASE_DIR / "plans"
OUTPUT_DIR = BASE_DIR / "output"
SETTINGS_FILE = BASE_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "confidence_threshold": 0.70,   # darunter: needs_review = true
    "csv_locale": "de",             # "de" = Semikolon + Dezimalkomma (deutsches Excel), "en" = Komma + Punkt
    "overlay_max_side": 4000,       # Größe des Overlay-Bildes (längste Seite, Pixel)
    "detector": {},                 # optionale Änderungen an CONFIG in stair_detector.py
    "method": "rules",              # "rules", "yolo" oder "both"
    "yolo": dict(yolo_detector.DEFAULT_YOLO),
}
METHODS = {"rules": "Rules", "yolo": "YOLO", "both": "Rules + YOLO"}

COLOR_OK = (0, 0, 255)              # rot (BGR) = Regeln
COLOR_REVIEW = (0, 140, 255)        # orange = Regeln, needs_review
COLOR_YOLO = (220, 90, 0)           # blau = YOLO
COLOR_YOLO_REVIEW = (230, 200, 0)   # hellblau = YOLO, needs_review
STAIR_TYPES = ["straight", "L-shaped", "U-shaped", "other"]
THUMB_SIZE = 700                    # Vorschaubild im Report (längste Seite, Pixel)


# --------------------------------------------------------------------------- Einstellungen
def load_settings():
    """Liest settings.json neben dem Programm. Fehlt die Datei, wird sie mit Standardwerten angelegt."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass                                        # kaputte Datei -> Standardwerte
    else:
        try:
            SETTINGS_FILE.write_text(json.dumps(DEFAULT_SETTINGS, indent=2), encoding="utf-8")
        except OSError:
            pass
    for key, value in settings.get("detector", {}).items():
        if key in sd.CONFIG:
            sd.CONFIG[key] = value
    yolo = dict(yolo_detector.DEFAULT_YOLO)
    yolo.update(settings.get("yolo") or {})
    settings["yolo"] = yolo
    if settings.get("method") not in METHODS:
        settings["method"] = "rules"
    return settings


def check_method(settings):
    """Wirft RuntimeError mit verständlicher Meldung, wenn YOLO gewählt, aber nicht bereit ist."""
    if settings.get("method", "rules") in ("yolo", "both"):
        ok, message = yolo_detector.status(settings)
        if not ok:
            raise RuntimeError(message)


def _label(base, settings):
    method = settings.get("method", "rules")
    return base if method == "rules" else f"{base}_{method}"


# --------------------------------------------------------------------------- Pläne finden
def list_plans():
    """Alle Pläne in plans/ (sortiert, ohne Unterordner)."""
    if not PLANS_DIR.exists():
        return []
    return sorted((p for p in PLANS_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in sd.IMAGE_TYPES),
                  key=lambda p: p.name.lower())


def find_plan(query):
    """
    Sucht einen Plan nach Namen: Groß-/Kleinschreibung egal, Dateiendung optional.
    Rückgabe: Path oder None.
    """
    q = query.strip().lower()
    if not q:
        return None
    for p in list_plans():
        if p.name.lower() == q or p.stem.lower() == q:
            return p
    return None


def filter_plans(text):
    """Für das Dropdown: alle Pläne, deren Name den Text enthält (Groß-/Kleinschreibung egal)."""
    t = text.strip().lower()
    return [p.name for p in list_plans() if t in p.name.lower()]


# --------------------------------------------------------------------------- Dateinamen/Ordner
def safe_name(name):
    """Ersetzt Zeichen, die in Windows-Dateinamen nicht erlaubt sind."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned or "plan"


def new_run_folder(label):
    """output/2026-09-30_14-32_<label>/ - existiert er schon, wird _2, _3 ... angehängt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now():%Y-%m-%d_%H-%M}_{safe_name(label)}"
    folder, n = OUTPUT_DIR / base, 2
    while folder.exists():
        folder, n = OUTPUT_DIR / f"{base}_{n}", n + 1
    folder.mkdir(parents=True)
    return folder


# --------------------------------------------------------------------------- Ergebnis eines Plans
def process_plan(path, out_dir, settings):
    """
    Analysiert einen Plan und schreibt <plan>_overlay.png und <plan>_result.json nach out_dir.
    Rückgabe: dict mit den Ergebnissen. Wirft eine Exception, wenn der Plan nicht lesbar ist.
    Fehler beim Speichern werden als 'save_error' / 'image_error' zurückgegeben.
    """
    path = Path(path)
    plan_name = path.stem
    method = settings.get("method", "rules")
    pages = sd.analyze_plan(path)                            # Regeln (auch für den Treppentyp bei YOLO)
    threshold = float(settings["confidence_threshold"])
    yolo_threshold = float(settings["yolo"]["review_threshold"])
    for page in pages:
        for s in page["stairs"]:
            s["method"] = "rules"
        if method in ("yolo", "both"):
            yolo_stairs = yolo_detector.detect_page(page, settings)
            page["stairs"] = yolo_stairs if method == "yolo" else page["stairs"] + yolo_stairs

    detections, det_id = [], 0
    for page in pages:
        for s in page["stairs"]:
            det_id += 1
            x1, y1, x2, y2 = s["bbox_xyxy"]
            s["id"] = det_id
            limit = yolo_threshold if s["method"] == "yolo" else threshold
            s["needs_review"] = s["confidence"] < limit
            detections.append({
                "id": det_id,
                "view": s["view"],
                "stair_type": s["stair_type"],
                "confidence": round(s["confidence"], 2),
                "needs_review": s["needs_review"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],          # [x, y, Breite, Höhe] im Original-TIF
                "page": page["page"] + 1,
                "staircase_id": s["staircase_id"],
                "n_treads": s["n_treads"],
                "method": s["method"],
            })

    result = {
        "plan_name": plan_name,
        "total_stairs_found": len(detections),
        "detections": detections,
        "method": method,
        "stairs_by_method": {m: sum(d["method"] == m for d in detections)
                             for m in (("rules", "yolo") if method == "both" else (method,))},
        "file_name": path.name,
        "image_size": list(pages[0]["original_size"]),
        "pages": len(pages),
        "program_version": PROGRAM_VERSION,
        "detector_version": sd.DETECTOR_VERSION,
        "confidence_threshold": threshold,
        "yolo_review_threshold": yolo_threshold if method != "rules" else None,
        "yolo_model": yolo_detector.model_path(settings).name if method != "rules" else None,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    stem = safe_name(plan_name)
    json_path = out_dir / f"{stem}_result.json"
    overlay_paths = []
    info = {"plan_name": plan_name, "result": result, "json_path": json_path,
            "overlay_paths": overlay_paths, "save_error": None, "image_error": None}
    try:
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        info["save_error"] = str(e)
    for page in pages:
        suffix = "" if page["page"] == 0 else f"_page{page['page'] + 1}"
        overlay_path = out_dir / f"{stem}{suffix}_overlay.png"
        try:
            write_overlay(page, overlay_path, settings)
            overlay_paths.append(overlay_path)
        except Exception as e:                               # noqa: BLE001
            info["image_error"] = str(e)
    return info


def write_overlay(page, overlay_path, settings):
    """Plan mit roter Box um jede Treppe (orange = needs_review). Bild ist verkleinert."""
    gray = page["gray"]
    factor = min(1.0, settings["overlay_max_side"] / max(gray.shape))
    if factor < 1.0:
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    to_img = page["scale"] * factor                          # Original -> Overlay
    thickness = max(2, int(max(img.shape) / 1200))
    font = max(0.5, max(img.shape) / 5000)
    for s in page["stairs"]:
        x1, y1, x2, y2 = [int(round(v * to_img)) for v in s["bbox_xyxy"]]
        yolo = s.get("method") == "yolo"
        if yolo:
            color = COLOR_YOLO_REVIEW if s["needs_review"] else COLOR_YOLO
        else:
            color = COLOR_REVIEW if s["needs_review"] else COLOR_OK
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = (f'#{s["id"]} {"YOLO " if yolo else ""}{s["confidence"]:.2f}'
                 + (" ?" if s["needs_review"] else ""))
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, font, color,
                    max(1, thickness // 2), cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise OSError("PNG could not be created")
    Path(overlay_path).write_bytes(buffer.tobytes())
    # kleines Vorschaubild für den HTML-Report
    t = min(1.0, THUMB_SIZE / max(img.shape))
    thumb = cv2.resize(img, None, fx=t, fy=t, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        thumb_path = Path(str(overlay_path).replace("_overlay.png", "_thumb.jpg"))
        thumb_path.write_bytes(buffer.tobytes())


# --------------------------------------------------------------------------- CSV
def _csv_format(settings):
    return (";", ",") if settings.get("csv_locale", "de") == "de" else (",", ".")


def _num(value, decimal):
    return f"{value:.2f}".replace(".", decimal)


def write_all_detections(run_dir, infos, settings):
    delim, dec = _csv_format(settings)
    with open(run_dir / "all_detections.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=delim)
        w.writerow(["plan_name", "detection_id", "view", "stair_type", "confidence",
                    "x", "y", "width", "height", "needs_review", "page", "staircase_id", "method"])
        for info in infos:
            for d in info["result"]["detections"]:
                w.writerow([info["plan_name"], d["id"], d["view"], d["stair_type"] or "",
                            _num(d["confidence"], dec), *d["bbox"],
                            "true" if d["needs_review"] else "false", d["page"], d["staircase_id"],
                            d.get("method", "rules")])


def write_summary(run_dir, rows, settings):
    delim, _ = _csv_format(settings)
    with open(run_dir / "summary_all_plans.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=delim)
        w.writerow(["plan_name", "total_stairs", *STAIR_TYPES, "stair_section",
                    "needs_review", "status", "method", "rules_stairs", "yolo_stairs"])
        for r in rows:
            w.writerow([r["plan_name"], r["total"], *[r["types"].get(t, 0) for t in STAIR_TYPES],
                        r["sections"], r["review"], r["status"], settings.get("method", "rules"),
                        r.get("rules", ""), r.get("yolo", "")])


def summary_row(info):
    dets = info["result"]["detections"]
    types = {}
    for d in dets:
        if d["view"] == "stair_plan":
            types[d["stair_type"]] = types.get(d["stair_type"], 0) + 1
    return {"plan_name": info["plan_name"], "total": len(dets), "types": types,
            "sections": sum(d["view"] == "stair_section" for d in dets),
            "review": sum(d["needs_review"] for d in dets), "status": "ok",
            "rules": sum(d.get("method") == "rules" for d in dets),
            "yolo": sum(d.get("method") == "yolo" for d in dets)}


# --------------------------------------------------------------------------- HTML-Report
def report_entry(info=None, plan=None, error=None):
    if info is None:
        return {"plan_name": plan.stem, "status": "error", "error": error, "result": None,
                "overlay": None, "thumb": None}
    overlay = info["overlay_paths"][0] if info["overlay_paths"] else None
    thumb = Path(str(overlay).replace("_overlay.png", "_thumb.jpg")) if overlay else None
    return {"plan_name": info["plan_name"], "status": "ok", "error": None, "result": info["result"],
            "overlay": overlay, "thumb": thumb if thumb and thumb.exists() else None}


def write_report(run_dir, label, entries, settings):
    """report.html im Lauf-Ordner. Ein Fehler hier stoppt den Lauf nicht."""
    try:
        return report.write_report(run_dir, label, entries, settings,
                                   PROGRAM_VERSION, sd.DETECTOR_VERSION)
    except Exception:                                        # noqa: BLE001
        return None


# --------------------------------------------------------------------------- Einzelplan
def run_single(plan_path, settings=None):
    """
    Verarbeitet einen Plan. Rückgabe: dict mit run_dir, overlay (Pfad) und Fehlerangaben.
    Wirft eine Exception, wenn der Plan selbst nicht verarbeitet werden kann.
    """
    settings = settings or load_settings()
    check_method(settings)
    plan_path = Path(plan_path)
    run_dir = new_run_folder(_label(plan_path.stem, settings))
    info = process_plan(plan_path, run_dir, settings)
    try:
        write_all_detections(run_dir, [info], settings)
    except OSError as e:
        info["save_error"] = info["save_error"] or str(e)
    info["run_dir"] = run_dir
    info["report_path"] = write_report(run_dir, f"Single plan: {plan_path.stem} · "
                                       f"{METHODS[settings.get('method', 'rules')]}",
                                       [report_entry(info)], settings)
    return info


# --------------------------------------------------------------------------- Batch
def run_batch(plan_paths=None, settings=None, progress=None, cancelled=None):
    """
    Verarbeitet alle Pläne nacheinander. progress(i, n, name) wird vor jedem Plan aufgerufen,
    cancelled() -> True bricht nach dem aktuellen Plan ab.
    Rückgabe: dict mit run_dir, ok, failed, total, cancelled.
    """
    settings = settings or load_settings()
    check_method(settings)
    plans = list(plan_paths) if plan_paths is not None else list_plans()
    run_dir = new_run_folder(_label(f"batch_{len(plans)}-plans", settings))
    log_lines = [f"Stairway to Heaven {PROGRAM_VERSION} (detector {sd.DETECTOR_VERSION})",
                 f"Method: {METHODS[settings.get('method', 'rules')]}",
                 f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}", f"Plans: {len(plans)}", ""]
    infos, rows, entries, ok, failed, was_cancelled = [], [], [], 0, 0, False
    for i, plan in enumerate(plans, start=1):
        if cancelled and cancelled():
            was_cancelled = True
            log_lines.append(f"Cancelled before plan {i} of {len(plans)}.")
            break
        if progress:
            progress(i, len(plans), plan.name)
        plan_dir = run_dir / safe_name(plan.stem)
        try:
            plan_dir.mkdir(exist_ok=True)
            info = process_plan(plan, plan_dir, settings)
            infos.append(info)
            rows.append(summary_row(info))
            entries.append(report_entry(info))
            ok += 1
            status = f"ok ({info['result']['total_stairs_found']} staircase flights)"
            if info["save_error"] or info["image_error"]:
                status += f" - warning: {info['save_error'] or info['image_error']}"
            log_lines.append(f"[{i}/{len(plans)}] {plan.name}: {status}")
        except Exception:                                    # noqa: BLE001 - Batch soll weiterlaufen
            failed += 1
            rows.append({"plan_name": plan.stem, "total": 0, "types": {}, "sections": 0,
                         "review": 0, "status": "error"})
            log_lines.append(f"[{i}/{len(plans)}] {plan.name}: ERROR")
            entries.append(report_entry(plan=plan, error=traceback.format_exc(limit=1).strip().splitlines()[-1]))
            try:
                plan_dir.rmdir()                             # leeren Ordner wieder entfernen
            except OSError:
                pass
            log_lines.append(traceback.format_exc())
    log_lines += ["", f"End: {datetime.now():%Y-%m-%d %H:%M:%S}",
                  f"Successful: {ok}, failed: {failed}"]
    try:
        write_all_detections(run_dir, infos, settings)
        write_summary(run_dir, rows, settings)
    except OSError as e:
        log_lines.append(f"CSV could not be saved: {e}")
    (run_dir / "processing_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    report_path = write_report(run_dir, f"Batch: {len(plans)} plans · "
                                        f"{METHODS[settings.get('method', 'rules')]}", entries, settings)
    return {"run_dir": run_dir, "ok": ok, "failed": failed, "total": len(plans),
            "cancelled": was_cancelled, "report_path": report_path}


# --------------------------------------------------------------------------- Terminal
def cli_main(args):
    """
    python main.py [plan ...] [--method rules|yolo|both]
    ohne Fenster; ohne Plan-Angabe werden alle Pläne verarbeitet.
    """
    settings = load_settings()
    args = list(args)
    if "--method" in args:
        i = args.index("--method")
        value = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
        if value not in METHODS:
            print("--method must be 'rules', 'yolo' or 'both'.")
            return
        settings["method"] = value
    try:
        check_method(settings)
    except RuntimeError as e:
        print(e)
        return
    print(f"Method: {METHODS[settings['method']]}")
    names = [a for a in args if not a.startswith("--")]
    if names:
        plans = []
        for n in names:
            p = find_plan(n) or (Path(n) if Path(n).exists() else None)
            if p is None:
                print(f"The plan '{n}' is not available in the folder. "
                      f"Please add the plan or choose a different one.")
            else:
                plans.append(p)
    else:
        plans = list_plans()
    if not plans:
        print(f"No plans found. Please add TIF files to {PLANS_DIR}.")
        return

    def show(i, n, name):
        print(f"Processing plan {i} of {n}: {name}")

    summary = run_batch(plans, settings, progress=show)
    print(f"Batch processing finished: {summary['ok']} of {summary['total']} plans processed "
          f"successfully, {summary['failed']} failed. See processing_log.txt for details.")
    print(f"Results: {summary['run_dir']}")
    if summary.get("report_path"):
        print(f"Report:  {summary['report_path']}")


if __name__ == "__main__":
    cli_main(sys.argv[1:])
