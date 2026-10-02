"""
Stairway to Heaven - Beispielbilder (data/examples/)

Hier legt ihr Ausschnitte/Screenshots/Scans von Treppen ab, die das Programm erkennen soll.
Der UNTERORDNER sagt, was auf dem Bild ist:

    data/examples/gerade/          gerade Treppe (Draufsicht)
    data/examples/l_form/          L-förmige Treppe
    data/examples/u_form/          U-förmige Treppe (zwei Läufe mit Podest)
    data/examples/wendel/          Wendel-/Spindeltreppe
    data/examples/rolltreppe/      Rolltreppe
    data/examples/sonstige/        andere Treppen (z. B. mehrläufig, gebogen)
    data/examples/seitenansicht/   Treppe im Schnitt/Seitenansicht (Zickzack-Profil)
    data/examples/keine_treppe/    sieht ähnlich aus, ist aber KEINE Treppe
                                   (Bohlenwand, Schraffur, Tabelle, Geländer ...)

Benutzt von:
    tools/check_examples.py  -> prüft, welche Beispiele Regeln/YOLO erkennen (Bericht)
    tools/prelabel.py        -> Box-Vorschläge zum Korrigieren in LabelImg
    tools/yolo_train.py      -> Beispiele als zusätzliche Trainingsbilder für YOLO

Der Maßstab eines Screenshots ist unbekannt (Zoomstufe!). Deshalb wird jedes Beispiel in
mehreren Größen geprüft; die Größe, in der die Regeln die Treppe am sichersten finden,
gilt als Maßstab (ungefähr 100 DPI wie bei den Plänen).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import stair_detector as sd

EXAMPLES_DIR = sd.PROJECT_DIR / "data" / "examples"

# Unterordner -> (erwartete Ansicht, erwarteter Treppentyp). None = keine Treppe
FOLDERS = {
    "gerade": ("stair_plan", "straight"),
    "l_form": ("stair_plan", "L-shaped"),
    "u_form": ("stair_plan", "U-shaped"),
    "wendel": ("stair_plan", "spiral"),
    "rolltreppe": ("stair_plan", "escalator"),
    "sonstige": ("stair_plan", "other"),
    "seitenansicht": ("stair_section", None),
    "keine_treppe": (None, None),
}
SCALES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MAX_SIDE = 4000                     # größere Vergrößerungen werden übersprungen


def create_folders():
    """Legt data/examples/ mit allen Unterordnern an (falls nicht vorhanden)."""
    for name in FOLDERS:
        (EXAMPLES_DIR / name).mkdir(parents=True, exist_ok=True)


def list_examples():
    """Alle Beispielbilder: Liste von (Ordnername, Pfad)."""
    items = []
    for folder in FOLDERS:
        d = EXAMPLES_DIR / folder
        if d.is_dir():
            for p in sorted(d.iterdir(), key=lambda p: p.name.lower()):
                if p.is_file() and p.suffix.lower() in sd.IMAGE_TYPES:
                    items.append((folder, p))
    return items


def load_gray(path):
    """Beispielbild als Graustufen-Array (transparente Bereiche werden weiß, DPI wird ignoriert)."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return np.array(img.convert("L"))


def resize(gray, scale):
    if scale == 1.0:
        return gray
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)


def usable_scales(gray):
    return [s for s in SCALES if max(gray.shape) * s <= MAX_SIDE] or [min(SCALES)]


def rules_best(gray, folder):
    """
    Regeln in allen Größen ausprobieren. Rückgabe: dict
        scale   : beste Größe (1.0, wenn nichts gefunden)
        stairs  : Treppen bei dieser Größe, bbox_xyxy in Pixeln des BEISPIELBILDES
        best    : sicherste Treppe der erwarteten Ansicht (oder None)
        any_scale_hits : {Größe: Anzahl Treppen} (für keine_treppe = Fehlalarme)
    """
    view, _ = FOLDERS.get(folder, ("stair_plan", None))
    best = {"scale": 1.0, "stairs": [], "best": None, "any_scale_hits": {}}
    best_key = (-1.0, 0.0)
    for s in usable_scales(gray):
        stairs = sd.analyze_array(resize(gray, s), name="", scale=s)
        best["any_scale_hits"][s] = len(stairs)
        wanted = [t for t in stairs if view is None or t["view"] == view]
        if not wanted:
            continue
        top = max(wanted, key=lambda t: t["confidence"])
        key = (top["confidence"], -abs(np.log(s)))          # bei Gleichstand: Größe nahe 1
        if key > best_key:
            best_key = key
            best.update(scale=s, stairs=stairs, best=top)
    return best


def to_yolo_lines(stairs, width, height, classes=("stair_plan", "stair_section")):
    """Treppen (bbox_xyxy in Pixeln) -> Zeilen im YOLO-Format (relativ 0..1)."""
    lines = []
    for t in stairs:
        if t["view"] not in classes:
            continue
        x1, y1, x2, y2 = t["bbox_xyxy"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{classes.index(t['view'])} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def read_yolo_boxes(path, width, height, keep_classes=(0, 1)):
    """YOLO-Datei -> Liste (klasse, x1, y1, x2, y2) in Pixeln. Fehlt die Datei: []."""
    boxes = []
    if not Path(path).exists():
        return boxes
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        if c in keep_classes:
            boxes.append((c, (cx - w / 2) * width, (cy - h / 2) * height,
                          (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes
