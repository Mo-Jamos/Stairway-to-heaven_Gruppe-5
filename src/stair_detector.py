"""
Stairway to Heaven - Treppenerkennung (Detektor v0.3, regelbasiert)

Erkennt Treppen in gescannten BVG-Plänen (TIF, PNG, JPG):
  - stair_plan    : Draufsicht = parallele Stufenlinien mit gleichen Abständen
                    und Lauflinie, in JEDEM Winkel (auch schief gescannt)
  - stair_section : Seitenansicht = Stufenprofil (Zickzack aus Auftritt + Steigung)
Eine Box pro Treppenlauf; zusammengehörige Läufe bekommen dieselbe staircase_id
und einen Treppentyp (straight, L-shaped, U-shaped, other).

Diese Datei enthält NUR die Erkennung. Ein-/Ausgabe (Ordner, JSON, CSV, Overlay):
src/stairway_app.py. Starten: python app_modern.py (Fenster) oder python main.py (Terminal).

Anforderungen (siehe PROJECT_BRIEF.md):
    REQ-1 Laden, auf 100 DPI bringen, binarisieren       -> load_image(), binarize()
    REQ-2 Rauschen entfernen                              -> remove_noise()
    REQ-3 Linien in allen häufigen Richtungen finden      -> dominant_angles(), axis_segments()
    REQ-4 Treppenkandidaten gruppieren                    -> group_into_staircases(), merge_flights()
    REQ-5 Fehlalarme filtern                              -> passes_filters(), is_grid(),
                                                             walkline_coverage(), too_much_text()
    REQ-6 Ergebnis liefern                                -> analyze_plan() (Ausgabe: stairway_app.py)
    REQ-7 Seitenansichten erkennen                        -> find_section_stairs()
    Treppentyp                                            -> assign_stair_types()

Versionen:
    v0.1  src/stair_detector_v01.py (nur 0°/90°, erste Version)
    v0.2  schiefe Treppen, unterbrochene Linien, Tabellen-/Textfilter, Seitenansicht, Lauflinie
    v0.3  ohne scikit-image (nur OpenCV), eine Box pro Lauf, Treppentyp, Konfidenz,
          mehrseitige TIFs, Schnittstelle analyze_plan() für Programm und .exe
    v0.4  Lauflinie mit gefüllter Pfeilspitze wird erkannt (remove_filled_blobs),
          analyze_array() für Ausschnitte/Screenshots (data/examples, YOLO-Treppentyp)
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Alle Einstellungen an einer Stelle. Pixelwerte beziehen sich auf das Bild
# NACH der Skalierung auf "target_dpi" (Standard: 100 DPI).
# Schwellwerte nur auf den Validierungsplänen einstellen, nie auf den Testplänen!
# ---------------------------------------------------------------------------
CONFIG = {
    # REQ-1: Auflösung
    "target_dpi": 100,        # alle Pläne auf diese Auflösung bringen (falls DPI bekannt)
    "max_side": 4000,         # Ersatz, falls der Scan keine DPI-Angabe hat
    "sauvola_window": 31,     # Fenstergröße der Sauvola-Binarisierung (ungerade Zahl)
    # REQ-2: Rauschen
    "min_blob_area": 30,      # kleinere Pixelgruppen gelten als Rauschen
    # REQ-3: Linien
    "min_line_length": 30,    # minimale Länge einer Stufenlinie
    "max_stroke": 6,          # dickere "Linien" sind Wände oder Schraffuren
    "hough_threshold": 20,    # Empfindlichkeit der Hough-Transformation
    "hough_max_gap": 4,       # Lücken bis zu so vielen Pixeln innerhalb einer Linie überbrücken
    "join_gap": 60,           # unterbrochene Stufenlinien (Text, Diagonale) wieder verbinden
    "angle_tolerance": 2,     # Richtungen, die sich um mehr als so viele Grad unterscheiden
    "min_angle_support": 400, # Summe der Linienlängen, ab der eine Richtung untersucht wird
    "max_angles": 8,          # höchstens so viele Richtungen pro Plan untersuchen
    "hatch_tolerance": 4,     # Richtungen um 45°/135° (typische Schraffur) überspringen
    # REQ-4: Gruppierung
    "min_overlap": 0.8,       # Linien müssen sich so stark überlappen
    "min_length_ratio": 0.7,  # kürzere / längere Linie
    "min_gap": 5,             # kleinster Abstand zwischen zwei Stufen
    "max_gap": 60,            # größter Abstand zwischen zwei Stufen
    "gap_tolerance": 0.35,    # erlaubte Abweichung eines Abstands vom Median
    "max_missing_treads": 1,  # so viele fehlende Stufenlinien hintereinander werden toleriert
    "merge_flights": True,    # Abschnitte derselben Treppe (Podest, Text) zusammenlegen
    "max_flight_distance": 2.5, # zwei vollständige Läufe: nur so kleine Lücken schließen
    "max_fragment_distance": 6, # Bruchstücke (< min_treads Linien) auch über größere Lücken verbinden
    "max_landing": 1.5,       # Podest: max. Abstand zweier Läufe derselben Treppe (x Treppenbreite)
    "max_eye": 0.6,           # Treppenauge: max. Abstand nebeneinanderliegender Läufe (x Breite)
    # REQ-5: Filter
    "min_treads": 5,          # Mindestanzahl Stufenlinien pro Treppe
    "max_gap_cv": 0.25,       # max. Variationskoeffizient der Abstände
    "min_aspect": 2.0,        # Stufenlänge / Stufenabstand (Schraffur < 2)
    "max_aspect": 25,         # Stufenlänge / Stufenabstand (Tabellenzeilen > 25)
    "max_grid_lines": 2,      # mehr durchgehende Querlinien = Tabelle/Raster, keine Treppe
    "min_walkline": 0.4,      # Lauflinie/Querlinie muss mind. 40 % der Treppentiefe abdecken
    "walkline_min_length": 15,# kürzeste Querlinie, die als Lauflinie zählt
    "arrow_blob_size": 7,     # gefüllte Pfeilspitzen (>= so dick) vor der Lauflinien-Suche entfernen
    "text_max_size": 25,      # Pixelgruppen bis zu dieser Größe gelten als Schriftzeichen
    "max_text_per_tread": 3,  # mehr Schriftzeichen pro Stufenlinie = Schriftfeld/Tabelle
    "min_box_size": 20,       # minimale Breite/Höhe einer Treppenbox
    "merge_overlap": 0.5,     # Boxen, die sich so stark überdecken, werden zusammengelegt
    "ignore_regions": {},     # z. B. {"1149-0007": [[x1, y1, x2, y2]]} in Originalpixeln
    # REQ-7: Seitenansicht
    "section_enabled": True,
    "section_min_step": 12,   # minimale Länge von Auftritt/Steigung
    "section_max_step": 90,   # maximale Länge von Auftritt/Steigung
    "section_touch": 5,       # so nah müssen sich Auftritt und Steigung berühren
    "section_min_steps": 3,   # Mindestanzahl Stufen im Profil
    "section_max_cv": 0.35,   # max. Variationskoeffizient der Stufenmaße
}

DETECTOR_VERSION = "0.4"
if getattr(sys, "frozen", False):                 # als .exe gestartet (PyInstaller)
    PROJECT_DIR = Path(sys.executable).resolve().parent
else:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
EXCLUDE_DIR = PROJECT_DIR / "data" / "exclude"   # "keine Treppe"-Boxen (LabelImg, YOLO-Format)
IMAGE_TYPES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

Image.MAX_IMAGE_PIXELS = None  # Baupläne sind oft sehr groß


def imwrite(path, image):
    """Bild speichern - funktioniert auch mit Umlauten im Dateinamen (cv2.imwrite unter Windows nicht)."""
    ok, buffer = cv2.imencode(".png", image)
    if ok:
        Path(path).write_bytes(buffer.tobytes())


# --------------------------------------------------------------------------- REQ-1
def get_scale(img):
    """Skalierungsfaktor: alle Pläne auf target_dpi bringen (z. B. 400 DPI -> 0.25)."""
    dpi = img.info.get("dpi")
    if dpi and dpi[0] and float(dpi[0]) > 1:
        scale = CONFIG["target_dpi"] / float(dpi[0])
    else:
        scale = CONFIG["max_side"] / max(img.size)
    return min(1.0, scale)


def count_pages(path):
    """Anzahl Seiten (mehrseitige TIFs)."""
    with Image.open(path) as img:
        return getattr(img, "n_frames", 1)


def load_image(path, page=0):
    """Lädt eine Seite eines Bildes (auch 1-Bit-TIF mit G4-Kompression) als Graustufen-Array."""
    img = Image.open(path)
    if page:
        img.seek(page)
    original_size = img.size
    scale = get_scale(img)
    gray = np.array(img.convert("L"))
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale, original_size


def threshold_sauvola(gray, window_size, k=0.2, r=127.5):
    """
    Sauvola-Schwellwert (lokaler Schwellwert pro Pixel), nur mit OpenCV:
    T = m * (1 + k * (s / r - 1)), m = lokaler Mittelwert, s = lokale Standardabweichung.
    """
    img = gray.astype(np.float32)
    size = (window_size, window_size)
    mean = cv2.boxFilter(img, -1, size, borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img * img, -1, size, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    return mean * (1 + k * (std / r - 1))


def skeletonize(mask, max_iter=12):
    """Morphologisches Skelett (1-2 Pixel breite Mittellinien), nur mit OpenCV."""
    img = mask.astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iter):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
        if not img.any():
            break
    return skel > 0


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
    Findet Liniensegmente in allen Richtungen (Skelett + Hough-Transformation).
    Wird nur benutzt, um die HÄUFIGEN RICHTUNGEN im Plan zu bestimmen
    (z. B. 0°, 90° und leicht schief gescannte 87°). Rückgabe: Array (x1, y1, x2, y2).
    """
    skeleton = skeletonize(ink > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 360, CONFIG["hough_threshold"],
                            minLineLength=CONFIG["min_line_length"],
                            maxLineGap=CONFIG["hough_max_gap"])
    if lines is None:
        return np.zeros((0, 4))
    return np.asarray(lines).reshape(-1, 4).astype(float)   # OpenCV 4 und 5 liefern unterschiedliche Formen


def dominant_angles(lines):
    """Welche Linienrichtungen kommen häufig vor? (Winkel in Grad, 0-180)"""
    peaks = [0, 90]                                   # waagrecht und senkrecht immer prüfen
    if len(lines) == 0:
        return peaks
    dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
    angles = np.degrees(np.arctan2(dy, dx)) % 180
    lengths = np.hypot(dx, dy)
    hist = np.zeros(180)
    for a, l in zip(angles, lengths):
        hist[int(round(a)) % 180] += l
    smooth = sum(np.roll(hist, k) for k in range(-1, 2))
    for a in np.argsort(smooth)[::-1]:
        if smooth[a] < CONFIG["min_angle_support"] or len(peaks) >= CONFIG["max_angles"]:
            break
        if any(min(abs(a - h), 180 - abs(a - h)) <= CONFIG["hatch_tolerance"] for h in (45, 135)):
            continue                                  # typische Schraffur-Winkel überspringen
        if all(min(abs(a - p), 180 - abs(a - p)) > CONFIG["angle_tolerance"] for p in peaks):
            peaks.append(int(a))
    return peaks


def rotate(ink, angle):
    """Dreht das Bild so, dass Linien mit Richtung 'angle' waagrecht liegen (ohne Abschneiden)."""
    h, w = ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    rotated = cv2.warpAffine(ink, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, cv2.invertAffineTransform(M)


def axis_segments(ink, horizontal=True, min_length=None):
    """
    Waagrechte (bzw. senkrechte) Linien per morphologischem Opening mit einem langen,
    flachen Strukturelement. Zu dicke Striche (Wände, Schraffuren) werden verworfen.
    Rückgabe: Liste (x1, y1, x2, y2).
    """
    n = min_length or CONFIG["min_line_length"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1) if horizontal else (1, n))
    mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        thick = h if horizontal else w
        if thick <= CONFIG["max_stroke"]:
            segs.append((x, y, x + w - 1, y + h - 1))
    return segs


def join_collinear(segs):
    """Verbindet Stücke derselben Linie (unterbrochen durch Text, Diagonale oder Scanlücken)."""
    items = sorted(((x1, (y1 + y2) / 2, x2, (y1 + y2) / 2) for x1, y1, x2, y2 in segs),
                   key=lambda s: s[1])
    rows = []
    for s in items:
        if rows and abs(rows[-1][0][1] - s[1]) <= 2:
            rows[-1].append(s)
        else:
            rows.append([s])
    joined = []
    for row in rows:
        row.sort(key=lambda s: s[0])
        cur = list(row[0])
        for s in row[1:]:
            if s[0] - cur[2] <= CONFIG["join_gap"]:
                cur[2] = max(cur[2], s[2])
            else:
                joined.append(tuple(cur))
                cur = list(s)
        joined.append(tuple(cur))
    return joined


def back_to_image(box, M_inv):
    """Box aus dem gedrehten Bild zurück ins Originalbild (achsparallele Hülle)."""
    u1, v1, u2, v2 = box
    pts = np.array([[u1, v1], [u2, v1], [u1, v2], [u2, v2]], dtype=float)
    xy = pts @ M_inv[:, :2].T + M_inv[:, 2]
    return [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]


# --------------------------------------------------------------------------- REQ-4
def _similar(a, b):
    """Überlappen sich zwei (waagrecht gedrehte) Segmente stark und sind ähnlich lang?"""
    len_a, len_b = a[2] - a[0], b[2] - b[0]
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    if overlap <= 0 or len_a <= 0 or len_b <= 0:
        return False
    shorter, longer = min(len_a, len_b), max(len_a, len_b)
    return overlap / shorter >= CONFIG["min_overlap"] and shorter / longer >= CONFIG["min_length_ratio"]


def _split_into_regular_runs(lines):
    """
    Teilt eine nach y sortierte Linienfolge in Abschnitte mit gleichmäßigen Abständen.
    Fehlt eine einzelne Stufenlinie (Abstand = 2 x normal), läuft der Abschnitt weiter.
    Rückgabe: Liste von (Linien, normierte Abstände, Anzahl ergänzter Stufen).
    """
    centers = [l[1] for l in lines]
    gaps = np.diff(centers)
    if len(gaps) == 0:
        return []
    median_gap = float(np.median(gaps))
    tol = CONFIG["gap_tolerance"]
    runs, current, norm, inferred = [], [lines[0]], [], 0
    for line, gap in zip(lines[1:], gaps):
        k = max(1, int(round(gap / median_gap)))
        if k <= CONFIG["max_missing_treads"] + 1 and abs(gap / k - median_gap) <= tol * median_gap:
            current.append(line)
            norm.append(gap / k)
            inferred += k - 1
        else:
            runs.append((current, norm, inferred))
            current, norm, inferred = [line], [], 0
    runs.append((current, norm, inferred))
    return runs


def group_into_staircases(segments):
    """
    Gruppiert ähnliche, parallele (gedrehte) Segmente zu Treppenkandidaten.
    Start bei den längsten Linien; von dort aus nach oben und unten weitersuchen.
    Rückgabe: Liste von dicts mit Box (im gedrehten System), Anzahl Stufen und Abständen.
    """
    segments = sorted(segments, key=lambda s: s[2] - s[0], reverse=True)
    used = [False] * len(segments)
    candidates = []
    for i, seed in enumerate(segments):
        if used[i]:
            continue
        similar = [j for j in range(len(segments))
                   if not used[j] and (j == i or _similar(seed, segments[j]))]
        similar.sort(key=lambda j: segments[j][1])
        pos = similar.index(i)
        chain = [i]
        for direction in (-1, +1):                   # vom Startpunkt nach oben und unten
            last = segments[i][1]
            k = pos + direction
            while 0 <= k < len(similar):
                gap = abs(segments[similar[k]][1] - last)
                if gap > CONFIG["max_gap"] * (CONFIG["max_missing_treads"] + 1):
                    break
                if gap >= CONFIG["min_gap"]:
                    chain.append(similar[k])
                    last = segments[similar[k]][1]
                k += direction
        for j in chain:
            used[j] = True
        lines = sorted((segments[j] for j in chain), key=lambda s: s[1])
        for run, norm_gaps, inferred in _split_into_regular_runs(lines):
            candidates.append({
                "frame_box": [min(l[0] for l in run), min(l[1] for l in run),
                              max(l[2] for l in run), max(l[3] for l in run)],
                "n_lines": len(run),
                "n_treads": len(run) + inferred,
                "gaps": np.array(norm_gaps),
                "tread_length": float(np.median([l[2] - l[0] for l in run])),
            })
    return candidates


def _max_distance(a, b):
    """
    Zwei vollständige Läufe (je >= min_treads Linien) bleiben getrennt (eine Box pro Lauf),
    wenn mehr als max_flight_distance Stufenabstände dazwischen liegen (Podest).
    Bruchstücke (z. B. durch Text unterbrochen) werden auch über größere Lücken verbunden.
    """
    if min(a["n_lines"], b["n_lines"]) < CONFIG["min_treads"]:
        return CONFIG["max_fragment_distance"]
    return CONFIG["max_flight_distance"]


def merge_flights(candidates):
    """
    Legt Abschnitte derselben Treppe zusammen (z. B. durch Podest, Text oder
    Lauflinie getrennt): gleiche Stufenlänge, gleicher Abstand, direkt übereinander.
    """
    candidates = sorted(candidates, key=lambda c: c["frame_box"][1])
    merged = []
    for c in candidates:
        for m in merged:
            a, b = m["frame_box"], c["frame_box"]
            overlap = min(a[2], b[2]) - max(a[0], b[0])
            shorter = min(a[2] - a[0], b[2] - b[0])
            ga, gb = np.mean(m["gaps"]), np.mean(c["gaps"])
            distance = b[1] - a[3]
            longer = max(a[2] - a[0], b[2] - b[0])
            if (shorter > 0 and overlap / shorter >= CONFIG["min_overlap"]
                    and shorter / longer >= CONFIG["min_length_ratio"]
                    and abs(ga - gb) <= CONFIG["gap_tolerance"] * min(ga, gb)
                    and -ga <= distance <= _max_distance(m, c) * min(ga, gb)):
                m["frame_box"] = [min(a[0], b[0]), a[1], max(a[2], b[2]), max(a[3], b[3])]
                m["n_treads"] += c["n_treads"]
                m["n_lines"] += c["n_lines"]
                m["gaps"] = np.concatenate([m["gaps"], c["gaps"]])
                break
        else:
            merged.append(c)
    return merged


# --------------------------------------------------------------------------- REQ-5
def passes_filters(c):
    """Verwirft Kandidaten, die wahrscheinlich keine Treppe sind."""
    if c["n_lines"] < CONFIG["min_treads"] or len(c["gaps"]) == 0:
        return False                                  # echte (nicht ergänzte) Linien zählen
    mean_gap = float(np.mean(c["gaps"]))
    cv = float(np.std(c["gaps"]) / mean_gap) if mean_gap > 0 else 1.0
    if cv > CONFIG["max_gap_cv"]:
        return False
    aspect = c["tread_length"] / mean_gap
    if aspect < CONFIG["min_aspect"] or aspect > CONFIG["max_aspect"]:
        return False
    c["score"] = round(max(0.0, 1.0 - cv) * min(1.0, c["n_treads"] / 10), 3)
    return True


def is_grid(c, cross_segments):
    """
    Tabelle/Raster statt Treppe? Eine Treppe hat höchstens wenige durchgehende Querlinien
    (Wangen, Lauflinie). Tabellen im Schriftfeld/Legende haben viele Spalten.
    """
    u1, v1, u2, v2 = c["frame_box"]
    width, height = u2 - u1, v2 - v1
    if height <= 0:
        return False
    count = 0
    for x1, y1, x2, y2 in cross_segments:          # senkrechte Linien im gedrehten Bild
        u = (x1 + x2) / 2
        if u1 + 0.1 * width < u < u2 - 0.1 * width:
            covered = min(y2, v2) - max(y1, v1)
            if covered >= 0.8 * height:
                count += 1
    return count > CONFIG["max_grid_lines"]


def _overlap(a, b):
    """Anteil der kleineren Box, der von der anderen überdeckt wird."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def merge_overlapping(dets):
    """Behält bei stark überlappenden Boxen nur die mit dem höchsten Score."""
    dets = sorted(dets, key=lambda d: (d["score"], d["n_treads"]), reverse=True)
    kept = []
    for d in dets:
        if all(_overlap(d["bbox"], k["bbox"]) < CONFIG["merge_overlap"] for k in kept):
            kept.append(d)
    return kept


def remove_filled_blobs(ink):
    """
    Entfernt gefüllte Flächen (z. B. Pfeilspitzen der Lauflinie). Grund: Pfeilspitze und
    Lauflinie bilden sonst EIN dickes Objekt, das als "zu dick für eine Linie" verworfen wird.
    Dünne Linien bleiben erhalten.
    """
    n = CONFIG["arrow_blob_size"]
    blobs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, n)))
    return cv2.subtract(ink, blobs)


def walkline_coverage(c, cross_segments):
    """
    Lauflinie: In Grundrissen wird eine Treppe mit einer Lauflinie (meist mit Pfeil)
    gezeichnet, die quer über die Stufen läuft. Rückgabe: größter Anteil der Treppentiefe,
    den eine Querlinie im Inneren abdeckt (0 = keine, 1 = über die ganze Treppe).
    Bohlenwände, Geländer, Kästchen usw. haben keine solche Linie.
    """
    u1, v1, u2, v2 = c["frame_box"]
    width, height = u2 - u1, v2 - v1
    if height <= 0:
        return 0.0
    best = 0.0
    for x1, y1, x2, y2 in cross_segments:
        u = (x1 + x2) / 2
        if u1 + 0.1 * width < u < u2 - 0.1 * width:
            best = max(best, (min(y2, v2) - max(y1, v1)) / height)
    return best


def plan_confidence(c, walk):
    """
    Regel-Konfidenz 0..1 (KEINE Wahrscheinlichkeit): gleichmäßige Abstände, genug Stufen,
    deutliche Lauflinie. Werte < confidence_threshold -> needs_review.
    """
    mean_gap = float(np.mean(c["gaps"]))
    cv = float(np.std(c["gaps"]) / mean_gap) if mean_gap > 0 else 1.0
    regular = max(0.0, 1.0 - cv)
    treads = min(1.0, c["n_lines"] / 8)
    walk_f = min(1.0, walk / 0.8)
    return round(regular * (0.5 + 0.5 * treads) * (0.6 + 0.4 * walk_f), 2)


def text_components(ink):
    """Kleine Pixelgruppen = meist Buchstaben/Ziffern. Rückgabe: Mittelpunkte."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.maximum(stats[1:, 2], stats[1:, 3]) <= CONFIG["text_max_size"]
    return centroids[1:][small]


def too_much_text(bbox, n_lines, text_centers):
    """Schriftfeld/Tabelle/Legende statt Treppe? Dann steht viel Text in der Box."""
    if len(text_centers) == 0:
        return False
    x1, y1, x2, y2 = bbox
    inside = ((text_centers[:, 0] > x1) & (text_centers[:, 0] < x2) &
              (text_centers[:, 1] > y1) & (text_centers[:, 1] < y2)).sum()
    return inside > CONFIG["max_text_per_tread"] * n_lines


def load_exclusions(name, image_shape):
    """
    "Das ist KEINE Treppe": Boxen, die ihr selbst festlegt.
    1. CONFIG["ignore_regions"]            -> Originalpixel [x1, y1, x2, y2]
    2. data/exclude/<plan>.txt (LabelImg)  -> YOLO-Format, relative Koordinaten 0..1
    Rückgabe: Boxen in Pixeln des verkleinerten Bildes.
    """
    h, w = image_shape
    boxes = []
    path = EXCLUDE_DIR / f"{name}.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def in_ignore_region(bbox, name, scale, exclusions):
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for x1, y1, x2, y2 in CONFIG["ignore_regions"].get(name, []):
        if x1 * scale <= cx <= x2 * scale and y1 * scale <= cy <= y2 * scale:
            return True
    for x1, y1, x2, y2 in exclusions:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


# --------------------------------------------------------------------------- Draufsicht
def detect_plan_stairs(ink):
    """Sucht Treppen in Draufsicht: für jede häufige Linienrichtung Bild drehen und suchen."""
    lines = find_line_segments(ink)
    text_centers = text_components(ink)
    thin = remove_filled_blobs(ink)                   # Lauflinie ohne Pfeilspitze
    detections = []
    for angle in dominant_angles(lines):
        rotated, M_inv = rotate(ink, angle)
        rotated_thin, _ = rotate(thin, angle)
        segs = join_collinear(axis_segments(rotated, horizontal=True))
        cross = axis_segments(rotated, horizontal=False)
        cross_short = (axis_segments(rotated, horizontal=False, min_length=CONFIG["walkline_min_length"])
                       + axis_segments(rotated_thin, horizontal=False,
                                       min_length=CONFIG["walkline_min_length"]))
        candidates = [c for c in group_into_staircases(segs) if len(c["gaps"]) >= 2]
        if CONFIG["merge_flights"]:
            candidates = merge_flights(candidates)
        for c in candidates:
            if not passes_filters(c) or is_grid(c, cross):
                continue
            walk = walkline_coverage(c, cross_short)
            if walk < CONFIG["min_walkline"]:
                continue                              # keine Lauflinie -> keine Treppe
            bbox = back_to_image(c["frame_box"], M_inv)
            if too_much_text(bbox, c["n_lines"], text_centers):
                continue
            if bbox[2] - bbox[0] < CONFIG["min_box_size"] or bbox[3] - bbox[1] < CONFIG["min_box_size"]:
                continue
            detections.append({"bbox": bbox, "class": "stair_plan", "n_treads": c["n_treads"],
                               "score": c["score"], "tread_angle": int(angle),
                               "walkline": round(float(walk), 2),
                               "confidence": plan_confidence(c, walk),
                               "tread_length": c["tread_length"],
                               "frame_box": c["frame_box"], "M_inv": M_inv})
    return detections, lines


# --------------------------------------------------------------------------- REQ-7
def _axis_segments(ink, horizontal):
    """Kurze waagrechte bzw. senkrechte Segmente für das Stufenprofil (Morphologie)."""
    n = CONFIG["section_min_step"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1) if horizontal else (1, n))
    mask = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        length, thick = (w, h) if horizontal else (h, w)
        if thick <= CONFIG["max_stroke"] + 2 and length <= CONFIG["section_max_step"]:
            segs.append((x, y, x + w - 1, y + h - 1))
    return segs


def find_section_stairs(ink):
    """
    Seitenansicht: Zickzack-Profil aus waagrechten Auftritten und senkrechten Steigungen,
    die sich an den Enden berühren und gleichmäßig groß sind.
    """
    tol = CONFIG["section_touch"]
    H = [(x1, (y1 + y2) / 2, x2) for x1, y1, x2, y2 in _axis_segments(ink, True)]
    V = [((x1 + x2) / 2, y1, y2) for x1, y1, x2, y2 in _axis_segments(ink, False)]
    H.sort(key=lambda h: h[0])
    results = []
    used = set()
    for start in range(len(H)):
        if start in used:
            continue
        for direction in (+1, -1):                   # +1 = nach rechts abwärts, -1 = aufwärts
            chain, risers, cur = [start], [], H[start]
            while True:
                nxt_v = None
                for v in V:
                    if abs(v[0] - cur[2]) > tol:
                        continue
                    top_hit, bottom_hit = abs(v[1] - cur[1]) <= tol, abs(v[2] - cur[1]) <= tol
                    if (direction == 1 and top_hit) or (direction == -1 and bottom_hit):
                        nxt_v = v
                        break
                if nxt_v is None:
                    break
                y_end = nxt_v[2] if direction == 1 else nxt_v[1]
                nxt_h = None
                for k, h in enumerate(H):
                    if abs(h[0] - nxt_v[0]) <= tol and abs(h[1] - y_end) <= tol and h[2] - h[0] > tol:
                        nxt_h = k
                        break
                if nxt_h is None or nxt_h in chain:
                    break
                risers.append(nxt_v[2] - nxt_v[1])
                chain.append(nxt_h)
                cur = H[nxt_h]
            if len(risers) >= CONFIG["section_min_steps"]:
                treads = [H[k][2] - H[k][0] for k in chain[1:-1]] or [H[chain[0]][2] - H[chain[0]][0]]
                cv_r = np.std(risers) / np.mean(risers)
                cv_t = np.std(treads) / np.mean(treads) if len(treads) > 1 else 0.0
                if cv_r <= CONFIG["section_max_cv"] and cv_t <= CONFIG["section_max_cv"]:
                    xs = [H[k][0] for k in chain] + [H[k][2] for k in chain]
                    ys = [H[k][1] for k in chain]
                    used.update(chain)
                    score = round(float(max(0.0, 1 - max(cv_r, cv_t))), 3)
                    results.append({"bbox": [min(xs), min(ys) - 2, max(xs), max(ys) + 2],
                                    "class": "stair_section", "n_treads": len(risers),
                                    "score": score,
                                    "confidence": round(score * min(1.0, len(risers) / 5), 2)})
    return results


# --------------------------------------------------------------------------- Treppentyp
def _flight_geometry(d):
    """Mittelpunkt, Laufrichtung, Stufenlänge und Lauflänge eines Treppenlaufs (Bildpixel)."""
    u1, v1, u2, v2 = d["frame_box"]
    M = d["M_inv"]
    center = np.array([(u1 + u2) / 2, (v1 + v2) / 2]) @ M[:, :2].T + M[:, 2]
    a = math.radians(d["tread_angle"])
    tread_dir = np.array([math.cos(a), math.sin(a)])      # entlang der Stufenlinien
    walk_dir = np.array([-math.sin(a), math.cos(a)])      # Laufrichtung
    return center, tread_dir, walk_dir, (u2 - u1), (v2 - v1)


def _relation(a, b):
    """Wie liegen zwei Läufe zueinander? 'straight' (hintereinander), 'U', 'L' oder None."""
    ca, ta, wa, la, da = _flight_geometry(a)
    cb, tb, wb, lb, db = _flight_geometry(b)
    diff = abs(a["tread_angle"] - b["tread_angle"]) % 180
    diff = min(diff, 180 - diff)
    delta = cb - ca
    width = max(la, lb)
    if diff <= 5:                                         # parallele Läufe
        along_tread = abs(delta @ ta)                     # seitlicher Versatz
        along_walk = abs(delta @ wa)                      # Versatz in Laufrichtung
        if along_tread < 0.5 * width and along_walk - (da + db) / 2 <= CONFIG["max_landing"] * width:
            return "straight"                             # hintereinander, Podest dazwischen
        if (along_tread - (la + lb) / 2 <= CONFIG["max_eye"] * width
                and along_walk < 0.5 * max(da, db) + 0.25 * width):
            return "U"                                    # nebeneinander (Treppenauge)
    elif diff >= 80:                                      # rechtwinklige Läufe
        gap = np.linalg.norm(delta) - (max(la, da) + max(lb, db)) / 2
        if gap <= CONFIG["max_landing"] * width:
            return "L"
    return None


def assign_stair_types(detections):
    """
    Eine Box pro Lauf. Läufe, die zusammengehören, bekommen dieselbe staircase_id.
    Treppentyp der Gruppe: straight (auch mit Podest), L-shaped, U-shaped oder other.
    Seitenansichten: stair_type = None.
    """
    plans = [d for d in detections if d["class"] == "stair_plan"]
    parent = list(range(len(plans)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    relations = {}
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            r = _relation(plans[i], plans[j])
            if r:
                parent[find(i)] = find(j)
                relations.setdefault(i, set()).add(r)
                relations.setdefault(j, set()).add(r)
    groups = {}
    for i in range(len(plans)):
        groups.setdefault(find(i), []).append(i)
    for gid, (_, members) in enumerate(sorted(groups.items()), start=1):
        rel = set().union(*(relations.get(i, set()) for i in members))
        if "L" in rel and "U" in rel:
            stair_type = "other"
        elif "L" in rel:
            stair_type = "L-shaped"
        elif "U" in rel:
            stair_type = "U-shaped"
        else:
            stair_type = "straight"
        for i in members:
            plans[i]["stair_type"] = stair_type
            plans[i]["staircase_id"] = gid
    next_id = len(groups) + 1
    for d in detections:
        if d["class"] == "stair_section":
            d["stair_type"] = None
            d["staircase_id"] = next_id
            next_id += 1
    return detections


# --------------------------------------------------------------------------- Pipeline
def detect(ink, name, scale):
    detections, lines = detect_plan_stairs(ink)
    if CONFIG["section_enabled"]:
        detections += find_section_stairs(ink)
    exclusions = load_exclusions(name, ink.shape)
    detections = [d for d in detections if not in_ignore_region(d["bbox"], name, scale, exclusions)]
    detections = merge_overlapping(detections)
    return assign_stair_types(detections), lines


def _to_stairs(detections, scale):
    """Interne Erkennungen -> Ergebnisliste in ORIGINAL-Pixeln (Bildpixel / scale)."""
    stairs = []
    for d in sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0])):
        x1, y1, x2, y2 = [v / scale for v in d["bbox"]]
        stairs.append({
            "view": d["class"],
            "stair_type": d.get("stair_type"),
            "staircase_id": d.get("staircase_id"),
            "confidence": float(d.get("confidence", d["score"])),
            "bbox_xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "n_treads": int(d["n_treads"]),
        })
    return stairs


def analyze_array(gray, name="", scale=1.0):
    """
    Erkennung auf einem Graustufen-Array, das schon im richtigen Maßstab (ca. 100 DPI) ist,
    z. B. ein Ausschnitt/Screenshot aus data/examples oder eine YOLO-Box.
    Rückgabe: Treppenliste in Pixeln des übergebenen Arrays / scale.
    """
    clean = remove_noise(binarize(gray))
    detections, _ = detect(clean, name, scale)
    return _to_stairs(detections, scale)


def analyze_page(path, page=0):
    """
    Analysiert eine Seite eines Plans. Rückgabe: dict mit dem verkleinerten Graustufenbild
    (für das Overlay), Skalierung, Originalgröße und den Treppen in ORIGINAL-Pixeln.
    """
    gray, scale, original_size = load_image(path, page)       # REQ-1
    binary = binarize(gray)                                    # REQ-1
    clean = remove_noise(binary)                               # REQ-2
    detections, lines = detect(clean, Path(path).stem, scale)  # REQ-3 bis REQ-5, REQ-7
    stairs = _to_stairs(detections, scale)
    return {"page": page, "gray": gray, "scale": scale, "original_size": original_size,
            "stairs": stairs, "binary": binary, "clean": clean, "lines": lines}


def analyze_plan(path):
    """Analysiert alle Seiten eines Plans (mehrseitige TIFs werden vollständig verarbeitet)."""
    return [analyze_page(path, p) for p in range(count_pages(path))]


if __name__ == "__main__":
    # Direkter Aufruf wie bisher: alle Pläne verarbeiten (Batch ohne Fenster)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stairway_app
    stairway_app.cli_main(sys.argv[1:])
