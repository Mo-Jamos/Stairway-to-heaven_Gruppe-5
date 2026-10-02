"""
Stairway to Heaven - Prototyp v0.1 (regelbasiert)

Erkennt Treppen in Draufsicht (parallele Stufenlinien mit gleichen Abständen)
in gescannten BVG-Plänen (TIF, PNG, JPG).

Aufruf (im VS-Code-Terminal, im Ordner stairway_prototype):
    python src/stair_detector_v01.py                  -> alle Dateien in plans/ (Ergebnis: outputs/v01)
    python src/stair_detector_v01.py plans/G_90_a.tif -> eine Datei
    python src/stair_detector.py --debug              -> zusätzlich Zwischenbilder speichern

Ausgabe in outputs/:
    <name>_overlay.png   Plan mit grünen Boxen um die gefundenen Treppen
    <name>.json          gefundene Treppen (Format mit Nerijus abgestimmt)
    <name>_1_binary.png, <name>_2_clean.png, <name>_3_lines.png  (nur mit --debug)

Anforderungen (siehe PROJECT_BRIEF.md):
    REQ-1 Laden und binarisieren      -> load_image(), binarize()
    REQ-2 Rauschen entfernen          -> remove_noise()
    REQ-3 Liniensegmente finden       -> find_line_segments()
    REQ-4 Treppenkandidaten gruppieren -> group_into_staircases()
    REQ-5 Fehlalarme filtern          -> passes_filters(), merge_overlapping()
    REQ-6 JSON und Overlay ausgeben   -> save_results()
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

# ---------------------------------------------------------------------------
# Alle Einstellungen an einer Stelle. Werte in Pixeln beziehen sich auf das
# Bild NACH der Skalierung mit "scale".
# Schwellwerte nur auf den Validierungsplänen einstellen, nie auf den Testplänen!
# ---------------------------------------------------------------------------
CONFIG = {
    "scale": "auto",          # "auto" = auf max_side verkleinern, oder feste Zahl, z. B. 0.25
    "max_side": 4000,         # bei "auto": längste Bildseite nach dem Verkleinern (Pixel)
    "sauvola_window": 31,     # Fenstergröße der Sauvola-Binarisierung (ungerade Zahl)
    "min_blob_area": 30,      # REQ-2: kleinere Pixelgruppen gelten als Rauschen
    "bridge_gap": 0,          # REQ-3: Lücken in Linien schließen, z. B. 3 (0 = aus)
    "min_line_length": 30,    # REQ-3: minimale Länge einer Stufenlinie
    "max_stroke": 6,          # REQ-3/5: dickere "Linien" sind Wände oder Schraffuren
    "min_overlap": 0.8,       # REQ-4: Linien müssen sich so stark überlappen
    "min_length_ratio": 0.7,  # REQ-4: kürzere / längere Linie
    "min_gap": 5,             # REQ-4: kleinster Abstand zwischen zwei Stufen
    "max_gap": 60,            # REQ-4: größter Abstand zwischen zwei Stufen
    "gap_tolerance": 0.35,    # REQ-4: erlaubte Abweichung eines Abstands vom Median
    "min_treads": 5,          # REQ-5: Mindestanzahl Stufenlinien pro Treppe
    "max_gap_cv": 0.25,       # REQ-5: max. Variationskoeffizient der Abstände
    "min_aspect": 4.0,        # REQ-5: Stufenlänge / Stufenabstand (Schraffur, Raster < 4)
    "min_box_size": 20,       # REQ-5: minimale Breite/Höhe einer Treppenbox
    "merge_iou": 0.3,         # REQ-5: überlappende Boxen zusammenführen
}

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "plans"
OUT_DIR = PROJECT_DIR / "outputs" / "v01"   # v0.1 getrennt ablegen, damit v0.2 nicht überschrieben wird
IMAGE_TYPES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

Image.MAX_IMAGE_PIXELS = None  # Baupläne sind oft sehr groß


def imwrite(path, image):
    """Bild speichern - funktioniert auch mit Umlauten im Dateinamen (cv2.imwrite unter Windows nicht)."""
    ok, buffer = cv2.imencode(".png", image)
    if ok:
        Path(path).write_bytes(buffer.tobytes())


# --------------------------------------------------------------------------- REQ-1
def get_scale(original_size):
    """Skalierungsfaktor: große Scans (z. B. 17000 x 8700 px bei 400 DPI) werden verkleinert."""
    if CONFIG["scale"] == "auto":
        return min(1.0, CONFIG["max_side"] / max(original_size))
    return float(CONFIG["scale"])


def load_image(path, scale):
    """Lädt ein Bild (auch 1-Bit-TIF mit G4-Kompression) als Graustufen-Array."""
    gray = np.array(Image.open(path).convert("L"))
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def binarize(gray):
    """Sauvola-Binarisierung. Ergebnis: Tinte = 255, Hintergrund = 0."""
    threshold = threshold_sauvola(gray, window_size=CONFIG["sauvola_window"])
    return ((gray < threshold) & (gray < 200)).astype(np.uint8) * 255


# --------------------------------------------------------------------------- REQ-2
def remove_noise(binary):
    """Entfernt kleine Pixelgruppen (Punktrauschen des Scans)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= CONFIG["min_blob_area"]
    return (keep[labels] * 255).astype(np.uint8)


# --------------------------------------------------------------------------- REQ-3
def find_line_segments(ink):
    """
    Findet waagrechte Liniensegmente per morphologischem Opening mit einem
    langen, flachen Strukturelement. Rückgabe: Liste von (x1, y1, x2, y2).
    Senkrechte Linien werden gefunden, indem man das Bild vorher transponiert.
    """
    if CONFIG["bridge_gap"] > 0:                # unterbrochene Scanlinien wieder verbinden
        bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (CONFIG["bridge_gap"], 1))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, bridge)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CONFIG["min_line_length"], 1))
    line_mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    segments = []
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        if h <= CONFIG["max_stroke"]:           # dicke Blöcke = Wände/Schraffur
            segments.append((x, y, x + w - 1, y + h - 1))
    return segments


# --------------------------------------------------------------------------- REQ-4
def _similar(a, b):
    """Überlappen sich zwei waagrechte Segmente stark und sind ähnlich lang?"""
    len_a, len_b = a[2] - a[0] + 1, b[2] - b[0] + 1
    overlap = min(a[2], b[2]) - max(a[0], b[0]) + 1
    if overlap <= 0:
        return False
    shorter, longer = min(len_a, len_b), max(len_a, len_b)
    return overlap / shorter >= CONFIG["min_overlap"] and shorter / longer >= CONFIG["min_length_ratio"]


def _split_into_regular_runs(lines):
    """Teilt eine nach y sortierte Linienfolge in Abschnitte mit gleichmäßigen Abständen."""
    centers = [(l[1] + l[3]) / 2 for l in lines]
    gaps = np.diff(centers)
    if len(gaps) == 0:
        return []
    median_gap = float(np.median(gaps))
    runs, current = [], [lines[0]]
    for line, gap in zip(lines[1:], gaps):
        if abs(gap - median_gap) <= CONFIG["gap_tolerance"] * median_gap:
            current.append(line)
        else:
            runs.append(current)
            current = [line]
    runs.append(current)
    return runs


def group_into_staircases(segments):
    """
    Gruppiert ähnliche, übereinander liegende Segmente zu Treppenkandidaten.
    Rückgabe: Liste von dicts mit bbox, Anzahl Stufen und Abständen.
    """
    segments = sorted(segments, key=lambda s: (s[1] + s[3]) / 2)
    used = [False] * len(segments)
    candidates = []
    for i, seed in enumerate(segments):
        if used[i]:
            continue
        chain, last = [seed], seed
        used[i] = True
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            gap = (segments[j][1] + segments[j][3]) / 2 - (last[1] + last[3]) / 2
            if gap > CONFIG["max_gap"]:
                break
            if gap >= CONFIG["min_gap"] and _similar(seed, segments[j]):
                chain.append(segments[j])
                used[j] = True
                last = segments[j]
        for run in _split_into_regular_runs(chain):
            centers = [(l[1] + l[3]) / 2 for l in run]
            gaps = np.diff(centers)
            candidates.append({
                "bbox": [min(l[0] for l in run), min(l[1] for l in run),
                         max(l[2] for l in run), max(l[3] for l in run)],
                "n_treads": len(run),
                "gaps": gaps,
                "tread_length": float(np.median([l[2] - l[0] + 1 for l in run])),
            })
    return candidates


# --------------------------------------------------------------------------- REQ-5
def passes_filters(c):
    """Verwirft Kandidaten, die wahrscheinlich keine Treppe sind."""
    if c["n_treads"] < CONFIG["min_treads"] or len(c["gaps"]) == 0:
        return False
    mean_gap = float(np.mean(c["gaps"]))
    cv = float(np.std(c["gaps"]) / mean_gap) if mean_gap > 0 else 1.0
    if cv > CONFIG["max_gap_cv"]:
        return False
    if c["tread_length"] / mean_gap < CONFIG["min_aspect"]:
        return False
    x1, y1, x2, y2 = c["bbox"]
    if x2 - x1 < CONFIG["min_box_size"] or y2 - y1 < CONFIG["min_box_size"]:
        return False
    c["score"] = round(max(0.0, 1.0 - cv) * min(1.0, c["n_treads"] / 10), 3)
    return True


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping(dets):
    """Behält bei stark überlappenden Boxen nur die mit dem höchsten Score."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        if all(_iou(d["bbox"], k["bbox"]) < CONFIG["merge_iou"] for k in kept):
            kept.append(d)
    return kept


# --------------------------------------------------------------------------- Pipeline
def detect(ink):
    """Sucht Treppen mit waagrechten UND senkrechten Stufenlinien."""
    detections = []
    for orientation in ("horizontal", "vertical"):
        img = ink if orientation == "horizontal" else np.ascontiguousarray(ink.T)
        for c in group_into_staircases(find_line_segments(img)):
            if passes_filters(c):
                x1, y1, x2, y2 = c["bbox"]
                bbox = [x1, y1, x2, y2] if orientation == "horizontal" else [y1, x1, y2, x2]
                detections.append({"bbox": [int(v) for v in bbox], "class": "stair_plan",
                                   "n_treads": c["n_treads"], "score": c["score"],
                                   "tread_direction": orientation})
    return merge_overlapping(detections)


# --------------------------------------------------------------------------- REQ-6
def save_results(path, gray, detections, original_size, s):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in detections:                        # zurück in Originalkoordinaten
        d["bbox"] = [int(round(v / s)) for v in d["bbox"]]
    result = {"image": path.name, "image_size": list(original_size), "scale_used": round(s, 4),
              "config": CONFIG, "detections": detections}
    (OUT_DIR / f"{path.stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thickness = max(2, int(max(gray.shape) / 1500))
    for d in detections:
        x1, y1, x2, y2 = [int(v * s) for v in d["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 170, 0), thickness)
        cv2.putText(overlay, f'{d["n_treads"]} / {d["score"]:.2f}', (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 130, 0), 1, cv2.LINE_AA)
    imwrite((OUT_DIR / f"{path.stem}_overlay.png"), overlay)


def save_debug(path, binary, clean):
    lines_img = cv2.cvtColor(255 - clean, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in find_line_segments(clean):
        cv2.rectangle(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for x1, y1, x2, y2 in find_line_segments(np.ascontiguousarray(clean.T)):
        cv2.rectangle(lines_img, (y1, x1), (y2, x2), (255, 0, 0), 1)
    imwrite((OUT_DIR / f"{path.stem}_1_binary.png"), 255 - binary)
    imwrite((OUT_DIR / f"{path.stem}_2_clean.png"), 255 - clean)
    imwrite((OUT_DIR / f"{path.stem}_3_lines.png"), lines_img)


def process(path, debug=False):
    original_size = Image.open(path).size           # (Breite, Höhe)
    scale = get_scale(original_size)
    gray = load_image(path, scale)                   # REQ-1
    binary = binarize(gray)                          # REQ-1
    clean = remove_noise(binary)                     # REQ-2
    detections = detect(clean)                       # REQ-3 bis REQ-5
    save_results(path, gray, detections, original_size, scale)  # REQ-6
    if debug:
        save_debug(path, binary, clean)
    print(f"{path.name}: {original_size[0]} x {original_size[1]} px, Faktor {scale:.2f} "
          f"-> {len(detections)} Treppe(n) gefunden")
    return detections


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    debug = "--debug" in sys.argv
    files = [Path(a) for a in args] or sorted(p for p in RAW_DIR.iterdir()
                                              if p.suffix.lower() in IMAGE_TYPES)
    if not files:
        print(f"Keine Bilder gefunden. Lege TIF-Dateien in {RAW_DIR} ab.")
        return
    for f in files:
        process(f, debug)
    print(f"Ergebnisse gespeichert in {OUT_DIR}")


if __name__ == "__main__":
    main()
