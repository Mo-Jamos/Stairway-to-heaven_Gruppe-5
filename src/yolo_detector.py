"""
Stairway to Heaven - Treppenerkennung mit YOLO (zweite Methode neben den Regeln)

YOLO lernt aus markierten Beispielen, wie Treppen aussehen (keine festen Regeln).
Voraussetzungen:
    1. python -m pip install ultralytics          (einmalig, installiert auch PyTorch)
    2. trainiertes Modell unter models/stairs_yolo.pt  (python tools/yolo_train.py)

Ablauf pro Planseite:
    Plan (schon auf 100 DPI gebracht) -> Kacheln 1024 x 1024 mit Überlappung
    -> YOLO auf jeder Kachel -> Boxen zurück in Plan-Koordinaten
    -> doppelte Boxen an Kachelrändern zusammenlegen
    -> Treppentyp (gerade/L/U) von der passenden Regel-Erkennung übernehmen

Klassen des Modells: 0 = stair_plan (Draufsicht), 1 = stair_section (Seitenansicht)
Einstellungen: settings.json -> "yolo": {...}
"""

from pathlib import Path

import cv2
import numpy as np

import stair_detector as sd

DEFAULT_YOLO = {
    "model": "models/stairs_yolo.pt",   # relativ zum Projektordner
    "min_confidence": 0.25,             # schwächere YOLO-Boxen werden verworfen
    "review_threshold": 0.50,           # darunter: needs_review = true
    "tile": 1024,                       # Kachelgröße in Pixeln (bei 100 DPI) - wie beim Training
    "overlap": 256,                     # Überlappung der Kacheln
    "type_from_rules": True,            # Treppentyp aus der Regel-Erkennung übernehmen
    "device": "",                       # "" = automatisch (Grafikkarte, sonst Prozessor)
}
CLASS_VIEWS = ["stair_plan", "stair_section"]
_MODELS = {}


def yolo_settings(settings):
    cfg = dict(DEFAULT_YOLO)
    cfg.update((settings or {}).get("yolo", {}))
    return cfg


def model_path(settings):
    p = Path(yolo_settings(settings)["model"])
    return p if p.is_absolute() else sd.PROJECT_DIR / p


def status(settings):
    """Kann YOLO benutzt werden? Rückgabe: (True/False, Meldung für das Fenster)."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False, ("YOLO is not installed. Run once in the terminal:\n"
                       "python -m pip install ultralytics")
    path = model_path(settings)
    if not path.exists():
        return False, (f"No YOLO model found:\n{path}\n\n"
                       "Annotate staircases and train first:\npython tools/yolo_train.py")
    return True, f"YOLO model: {path.name}"


def load_model(settings):
    path = model_path(settings)
    key = (str(path), path.stat().st_mtime)
    if key not in _MODELS:
        from ultralytics import YOLO
        _MODELS.clear()
        _MODELS[key] = YOLO(str(path))
    return _MODELS[key]


def tile_positions(length, tile, overlap):
    """Startpositionen der Kacheln entlang einer Seite (letzte Kachel bündig am Rand)."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    pos = list(range(0, length - tile, step))
    pos.append(length - tile)
    return pos


def make_tile(gray, x, y, tile):
    """Ausschnitt; zu kleine Ränder werden WEISS aufgefüllt (YOLO soll nicht vergrößern)."""
    crop = gray[y:y + tile, x:x + tile]
    if crop.shape != (tile, tile):
        canvas = np.full((tile, tile), 255, dtype=np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    return crop


def _overlap_smaller(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def merge_tile_boxes(boxes):
    """
    Dieselbe Treppe kann in zwei überlappenden Kacheln gefunden werden (ganz oder angeschnitten).
    - fast gleiche Boxen (IoU >= 0.6): nur die sicherste behalten
    - eine Box ist am Kachelrand ABGESCHNITTEN und liegt größtenteils in einer anderen Box
      aus einer anderen Kachel: beide zur Hülle zusammenlegen (aber nur, wenn die Hülle
      nicht viel größer wird - sonst würden benachbarte Treppen verschmelzen)
    """
    boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept = []
    for b in boxes:
        for k in kept:
            if k["view"] != b["view"]:
                continue
            if ((k["cut"] or b["cut"]) and k["tile"] != b["tile"]
                    and _overlap_smaller(k["bbox"], b["bbox"]) >= 0.5):
                hull = [min(k["bbox"][0], b["bbox"][0]), min(k["bbox"][1], b["bbox"][1]),
                        max(k["bbox"][2], b["bbox"][2]), max(k["bbox"][3], b["bbox"][3])]
                if _area(hull) <= 1.5 * max(_area(k["bbox"]), _area(b["bbox"])):
                    k["bbox"], k["cut"] = hull, False
                    break
            if _iou(k["bbox"], b["bbox"]) >= 0.6:
                break
        else:
            kept.append(b)
    return kept


def detect_gray(gray, settings, batch=8):
    """
    YOLO auf einem Graustufenbild (ca. 100 DPI). Rückgabe: Liste von dicts
    {bbox: [x1, y1, x2, y2] in Pixeln von gray, view, confidence}.
    """
    cfg = yolo_settings(settings)
    model = load_model(settings)
    names = getattr(model, "names", {}) or {}
    tile, overlap = int(cfg["tile"]), int(cfg["overlap"])
    h, w = gray.shape[:2]
    jobs = [(x, y) for y in tile_positions(h, tile, overlap) for x in tile_positions(w, tile, overlap)]
    boxes = []
    for start in range(0, len(jobs), batch):
        part = jobs[start:start + batch]
        images = [cv2.cvtColor(make_tile(gray, x, y, tile), cv2.COLOR_GRAY2BGR) for x, y in part]
        kwargs = {"imgsz": tile, "conf": float(cfg["min_confidence"]), "verbose": False}
        if cfg.get("device"):
            kwargs["device"] = cfg["device"]
        results = model.predict(images, **kwargs)
        for (x, y), res in zip(part, results):
            if res.boxes is None:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                name = names.get(int(k), "") if isinstance(names, dict) else ""
                view = name if name in CLASS_VIEWS else CLASS_VIEWS[min(int(k), 1)]
                bx = [float(x1 + x), float(y1 + y), float(min(x2 + x, w)), float(min(y2 + y, h))]
                # abgeschnitten = berührt einen Kachelrand, der NICHT der Bildrand ist
                cut = ((x1 <= 3 and x > 0) or (y1 <= 3 and y > 0)
                       or (x2 >= tile - 3 and x + tile < w) or (y2 >= tile - 3 and y + tile < h))
                if bx[2] - bx[0] >= 4 and bx[3] - bx[1] >= 4:
                    boxes.append({"bbox": bx, "view": view, "confidence": float(c),
                                  "tile": (x, y), "cut": bool(cut)})
    result = merge_tile_boxes(boxes)
    for b in result:
        b.pop("tile", None)
        b.pop("cut", None)
    return result


def detect_page(page, settings):
    """
    YOLO-Erkennung für eine Seite aus sd.analyze_page(). Rückgabe: Treppen im selben Format
    wie die Regeln (bbox_xyxy in ORIGINAL-Pixeln) plus "method": "yolo".
    Treppentyp, Treppen-ID und Stufenzahl kommen von der passenden Regel-Box (falls vorhanden).
    """
    cfg = yolo_settings(settings)
    scale = page["scale"]
    rule_stairs = [s for s in page["stairs"] if s.get("method", "rules") == "rules"]
    next_id = max([s.get("staircase_id") or 0 for s in rule_stairs] + [0]) + 1
    stairs = []
    for b in detect_gray(page["gray"], settings):
        x1, y1, x2, y2 = [v / scale for v in b["bbox"]]
        box = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        match = None
        if cfg["type_from_rules"]:
            cands = [(_iou(box, r["bbox_xyxy"]), r) for r in rule_stairs if r["view"] == b["view"]]
            cands = [(v, r) for v, r in cands if v >= 0.3 or _overlap_smaller(box, r["bbox_xyxy"]) >= 0.7]
            if cands:
                match = max(cands, key=lambda c: c[0])[1]
        if match:
            stair_type, sid, n_treads = match["stair_type"], match["staircase_id"], match["n_treads"]
        else:
            stair_type = "unknown" if b["view"] == "stair_plan" else None
            sid, n_treads = next_id, None
            next_id += 1
        stairs.append({"view": b["view"], "stair_type": stair_type, "staircase_id": sid,
                       "confidence": round(b["confidence"], 3), "bbox_xyxy": box,
                       "n_treads": n_treads, "method": "yolo"})
    stairs.sort(key=lambda s: (s["bbox_xyxy"][1], s["bbox_xyxy"][0]))
    return stairs
