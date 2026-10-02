"""
Box-Vorschläge zum Markieren (spart viel Zeit beim Annotieren)

Das Regel-Programm zeichnet die Treppen vor, ihr KORRIGIERT nur noch in LabelImg:
fehlende Treppen ergänzen, falsche Boxen löschen, Boxen enger ziehen.

    python tools/prelabel.py

Was passiert:
  1. Pläne (plans/):  data/annotate/<plan>.png      (verkleinerte Kopie zum Anschauen)
                      data/annotations/<plan>.txt   (Vorschläge im YOLO-Format)
  2. Beispiele (data/examples/<ordner>/<bild>):  <bild>.txt daneben
       keine_treppe/  -> leere .txt (= keine Treppe auf dem Bild, richtig so)

Sicher:
  - Vorhandene .txt-Dateien werden NIE überschrieben (eure Korrekturen bleiben erhalten).
  - Pläne aus data/split_test.txt bekommen KEINE Vorschläge: Testpläne immer selbst und
    unabhängig markieren, sonst ist der Vergleich Regeln vs. YOLO nicht fair.

Danach korrigieren (Klassen: 0 stair_plan, 1 stair_section, 2 hard_negative):
    python -m labelImg.labelImg data/annotate data/classes.txt data/annotations
    python -m labelImg.labelImg data/examples/gerade data/classes.txt data/examples/gerade
Fertig korrigierte Pläne in data/geprueft.txt eintragen (ein Planname pro Zeile).
"""
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import examples as ex          # noqa: E402
import stair_detector as sd    # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PROJECT_DIR = sd.PROJECT_DIR
PLANS_DIR = PROJECT_DIR / "plans"
ANNOTATE_DIR = PROJECT_DIR / "data" / "annotate"
ANN_DIR = PROJECT_DIR / "data" / "annotations"
CLASSES = PROJECT_DIR / "data" / "classes.txt"
PNG_MAX_SIDE = 4000


def read_list(path):
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")}


def copy_classes(folder):
    """LabelImg (YOLO-Modus) braucht classes.txt im Speicherordner."""
    target = folder / "classes.txt"
    if CLASSES.exists() and not target.exists():
        shutil.copyfile(CLASSES, target)


def prelabel_plans():
    ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    copy_classes(ANN_DIR)
    test_plans = read_list(PROJECT_DIR / "data" / "split_test.txt")
    plans = sorted(p for p in PLANS_DIR.iterdir() if p.is_file() and p.suffix.lower() in sd.IMAGE_TYPES)
    for plan in plans:
        png = ANNOTATE_DIR / f"{plan.stem}.png"
        if not png.exists():
            img = Image.open(plan).convert("L")
            s = min(1.0, PNG_MAX_SIDE / max(img.size))
            img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS).save(png)
        label = ANN_DIR / f"{plan.stem}.txt"
        if label.exists():
            print(f"  {plan.name}: Annotation vorhanden - bleibt unverändert")
            continue
        if plan.stem in test_plans:
            print(f"  {plan.name}: Testplan - KEINE Vorschläge, bitte selbst markieren")
            continue
        page = sd.analyze_page(plan, 0)
        w, h = page["original_size"]
        lines = ex.to_yolo_lines(page["stairs"], w, h)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"  {plan.name}: {len(lines)} Vorschläge -> {label.relative_to(PROJECT_DIR)}")


def prelabel_examples():
    ex.create_folders()
    items = ex.list_examples()
    if not items:
        print("  keine Beispielbilder in data/examples/")
        return
    todo = []
    for folder in ex.FOLDERS:
        copy_classes(ex.EXAMPLES_DIR / folder)
    for folder, path in items:
        label = path.with_suffix(".txt")
        if label.exists():
            continue
        if ex.FOLDERS[folder][0] is None:                     # keine_treppe
            label.write_text("", encoding="utf-8")
            continue
        gray = ex.load_gray(path)
        res = ex.rules_best(gray, folder)
        if res["best"] is None:
            todo.append(f"{folder}/{path.name}")
            continue
        h, w = gray.shape
        lines = ex.to_yolo_lines(res["stairs"], w, h)
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {folder}/{path.name}: {len(lines)} Vorschläge (Größe x{res['scale']})")
    if todo:
        print("\n  Diese Beispiele haben die Regeln NICHT erkannt - bitte in LabelImg selbst markieren:")
        for t in todo:
            print(f"    {t}")


def main():
    print("Pläne:")
    prelabel_plans()
    print("\nBeispiele (data/examples):")
    prelabel_examples()
    print("\nJetzt korrigieren mit LabelImg (Format: YOLO!):")
    print("  python -m labelImg.labelImg data/annotate data/classes.txt data/annotations")
    print("  python -m labelImg.labelImg data/examples/<ordner> data/classes.txt data/examples/<ordner>")
    print("Fertig korrigierte Pläne in data/geprueft.txt eintragen (ein Planname pro Zeile).")


if __name__ == "__main__":
    main()
