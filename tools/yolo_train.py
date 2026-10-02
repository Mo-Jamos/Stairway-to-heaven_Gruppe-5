"""
YOLO trainieren (Treppen erkennen lernen)

Voraussetzungen:
    python -m pip install ultralytics                      (einmalig)
    markierte Pläne: data/annotations/<plan>.txt           (tools/prelabel.py + LabelImg)
    fertig geprüfte Pläne in data/geprueft.txt eintragen   (ein Planname pro Zeile)
    optional: Beispiele mit .txt in data/examples/<ordner>/

Aufruf:
    python tools/yolo_train.py                    Datensatz bauen + trainieren (Laptop)
    python tools/yolo_train.py --epochs 150       länger trainieren
    python tools/yolo_train.py --zip              nur Datensatz als ZIP für Google Colab
    python tools/yolo_train.py --include-unchecked   auch ungeprüfte Vorschläge benutzen
                                                  (nur zum Ausprobieren - YOLO lernt dann
                                                  die Fehler der Regeln mit!)

Ablauf:
    1. Datensatz bauen: data/yolo/dataset/images|labels/train|val
         - jeder Plan wird wie bei der Erkennung auf 100 DPI gebracht und in Kacheln
           1024 x 1024 (Überlappung 256) geschnitten
         - Pläne aus data/split_val.txt  -> val (Kontrolle während des Trainings)
         - Pläne aus data/split_test.txt -> werden NIE benutzt (Endauswertung!)
         - Beispiele aus data/examples   -> train (keine_treppe = Gegenbeispiele)
         - jedes Trainingsbild zusätzlich um 90° gedreht (Treppen in allen Richtungen)
    2. Training mit einem vortrainierten Modell (Standard: yolo11n.pt, wird beim ersten Mal
       automatisch heruntergeladen). Mit Grafikkarte: Minuten, nur Prozessor: Stunden.
    3. bestes Modell -> models/stairs_yolo.pt (altes Modell wird mit Datum gesichert)

Danach im Fenster Methode "YOLO" oder "Beide vergleichen" wählen.
"""
import argparse
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import examples as ex          # noqa: E402
import stair_detector as sd    # noqa: E402
import stairway_app as core    # noqa: E402
import yolo_detector as yd     # noqa: E402

PROJECT_DIR = sd.PROJECT_DIR
ANN_DIR = PROJECT_DIR / "data" / "annotations"
YOLO_DIR = PROJECT_DIR / "data" / "yolo"
DATASET = YOLO_DIR / "dataset"
RUNS_DIR = YOLO_DIR / "runs"
MODELS_DIR = PROJECT_DIR / "models"
NAMES = ["stair_plan", "stair_section"]
MIN_VISIBLE = 0.4          # angeschnittene Treppe: Box behalten, wenn >= 40 % sichtbar
MIN_INK = 0.001            # fast leere (weiße) Kacheln überspringen


def read_list(path):
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")}


def tile_image(gray, boxes, tile, overlap):
    """
    Schneidet ein Bild in Kacheln. boxes: (klasse, x1, y1, x2, y2) in Pixeln von gray.
    Angeschnittene Treppen mit < 40 % Sichtbarkeit werden in der Kachel weiß übermalt,
    damit YOLO sie nicht als "keine Treppe" lernt.
    Rückgabe: Liste (x, y, kachel, zeilen im YOLO-Format)
    """
    h, w = gray.shape
    out = []
    for y in yd.tile_positions(h, tile, overlap):
        for x in yd.tile_positions(w, tile, overlap):
            img = yd.make_tile(gray, x, y, tile).copy()
            lines = []
            for c, x1, y1, x2, y2 in boxes:
                area = max(1e-6, (x2 - x1) * (y2 - y1))
                ix1, iy1 = max(x1, x), max(y1, y)
                ix2, iy2 = min(x2, x + tile), min(y2, y + tile)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                visible = (ix2 - ix1) * (iy2 - iy1) / area
                tx1, ty1, tx2, ty2 = ix1 - x, iy1 - y, ix2 - x, iy2 - y
                if visible >= MIN_VISIBLE:
                    lines.append(f"{c} {(tx1 + tx2) / 2 / tile:.6f} {(ty1 + ty2) / 2 / tile:.6f} "
                                 f"{(tx2 - tx1) / tile:.6f} {(ty2 - ty1) / tile:.6f}")
                else:
                    img[int(ty1):int(np.ceil(ty2)), int(tx1):int(np.ceil(tx2))] = 255
            if lines or (img < 128).mean() >= MIN_INK:
                out.append((x, y, img, lines))
    return out


def rotate90(img, lines):
    """Kachel um 90° im Uhrzeigersinn drehen, Boxen mitdrehen."""
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new = []
    for line in lines:
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        new.append(f"{c} {1 - cy:.6f} {cx:.6f} {h:.6f} {w:.6f}")
    return rot, new


def save(split, name, img, lines):
    for sub in ("images", "labels"):
        (DATASET / sub / split).mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return
    (DATASET / "images" / split / f"{name}.png").write_bytes(buf.tobytes())
    (DATASET / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                           encoding="utf-8")


def build_dataset(settings, include_unchecked=False, rot90=True):
    cfg = yd.yolo_settings(settings)
    tile, overlap = int(cfg["tile"]), int(cfg["overlap"])
    if DATASET.exists():
        shutil.rmtree(DATASET)
    val_plans = read_list(PROJECT_DIR / "data" / "split_val.txt")
    test_plans = read_list(PROJECT_DIR / "data" / "split_test.txt")
    checked = read_list(PROJECT_DIR / "data" / "geprueft.txt")
    stats = {"train": [0, 0], "val": [0, 0]}                 # [Kacheln, Treppen-Boxen]
    used, skipped = [], []

    def add(split, name, img, lines):
        save(split, name, img, lines)
        stats[split][0] += 1
        stats[split][1] += len(lines)
        if split == "train" and rot90:
            r_img, r_lines = rotate90(img, lines)
            save(split, name + "_r90", r_img, r_lines)
            stats[split][0] += 1
            stats[split][1] += len(r_lines)

    # --- Pläne
    for plan in core.list_plans():
        stem = plan.stem
        ann = ANN_DIR / f"{stem}.txt"
        if stem in test_plans:
            skipped.append(f"{plan.name}: Testplan (wird nie trainiert)")
            continue
        if not ann.exists():
            skipped.append(f"{plan.name}: keine Annotation (tools/prelabel.py)")
            continue
        if stem not in checked and not include_unchecked:
            skipped.append(f"{plan.name}: nicht in data/geprueft.txt")
            continue
        split = "val" if stem in val_plans else "train"
        gray, _, _ = sd.load_image(plan, 0)                  # Annotationen gelten für Seite 1
        h, w = gray.shape
        boxes = ex.read_yolo_boxes(ann, w, h)
        for x, y, img, lines in tile_image(gray, boxes, tile, overlap):
            add(split, f"{core.safe_name(stem)}__x{x}_y{y}", img, lines)
        used.append(f"{plan.name} -> {split}")

    # --- Beispiele
    for folder, path in ex.list_examples():
        label = path.with_suffix(".txt")
        is_negative = ex.FOLDERS[folder][0] is None
        if not label.exists() and not is_negative:
            skipped.append(f"Beispiel {folder}/{path.name}: keine .txt (tools/prelabel.py / LabelImg)")
            continue
        gray = ex.load_gray(path)
        res = ex.rules_best(gray, folder) if not is_negative else {"scale": 1.0}
        s = res["scale"]
        h0, w0 = gray.shape
        boxes = [] if is_negative else ex.read_yolo_boxes(label, w0 * s, h0 * s)
        g = ex.resize(gray, s)
        for x, y, img, lines in tile_image(g, boxes, tile, overlap):
            add("train", f"ex_{folder}__{core.safe_name(path.stem)}__x{x}_y{y}", img, lines)
        used.append(f"Beispiel {folder}/{path.name} (Größe x{s})")

    if stats["val"][0] == 0 and stats["train"][0] > 0:
        print("WARNUNG: Kein Validierungsplan (data/split_val.txt) markiert. Die Trainingsbilder werden\n"
              "         auch zur Kontrolle benutzt - die angezeigten Werte sind dann zu optimistisch.")
        for sub in ("images", "labels"):
            (DATASET / sub / "val").mkdir(parents=True, exist_ok=True)
        for f in (DATASET / "images" / "train").glob("*.png"):
            if "_r90" not in f.stem:
                shutil.copyfile(f, DATASET / "images" / "val" / f.name)
                shutil.copyfile(DATASET / "labels" / "train" / f"{f.stem}.txt",
                                DATASET / "labels" / "val" / f"{f.stem}.txt")
    yaml = DATASET / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    yaml.write_text(f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
                    encoding="utf-8")
    (DATASET / "data_colab.yaml").write_text(
        f"path: /content/dataset\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")

    print("Benutzt:")
    for u in used:
        print(f"  {u}")
    if skipped:
        print("Übersprungen:")
        for s in skipped:
            print(f"  {s}")
    print(f"Datensatz: train {stats['train'][0]} Kacheln / {stats['train'][1]} Treppen-Boxen, "
          f"val {stats['val'][0]} Kacheln / {stats['val'][1]} Boxen")
    if stats["train"][1] < 50:
        print("HINWEIS: Sehr wenige Treppen-Boxen. Für ein brauchbares Modell braucht ihr deutlich mehr\n"
              "         markierte Treppen (Richtwert: einige hundert aus vielen verschiedenen Plänen).")
    return yaml, stats


def make_zip():
    target = YOLO_DIR / "stairs_dataset.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for f in DATASET.rglob("*"):
            if f.is_file():
                z.write(f, Path("dataset") / f.relative_to(DATASET))
    return target


def install_model(best):
    MODELS_DIR.mkdir(exist_ok=True)
    target = MODELS_DIR / "stairs_yolo.pt"
    if target.exists():
        backup = MODELS_DIR / f"stairs_yolo_{datetime.now():%Y-%m-%d_%H-%M}.pt"
        shutil.copyfile(target, backup)
        print(f"Altes Modell gesichert: {backup.name}")
    shutil.copyfile(best, target)
    return target


def main():
    ap = argparse.ArgumentParser(description="YOLO-Treppenmodell trainieren")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model", default="yolo11n.pt", help="Startmodell (yolo11n.pt klein/schnell, yolo11s.pt genauer)")
    ap.add_argument("--batch", type=int, default=4, help="Bilder pro Schritt (Grafikkarte: 8-16)")
    ap.add_argument("--device", default="", help='"" automatisch, "cpu" oder "0" (erste Grafikkarte)')
    ap.add_argument("--zip", action="store_true", help="nur Datensatz-ZIP für Google Colab erzeugen")
    ap.add_argument("--include-unchecked", action="store_true")
    ap.add_argument("--no-rot90", action="store_true", help="keine um 90° gedrehten Kopien")
    ap.add_argument("--build-only", action="store_true", help="nur Datensatz bauen, nicht trainieren")
    args = ap.parse_args()

    settings = core.load_settings()
    yaml, stats = build_dataset(settings, args.include_unchecked, not args.no_rot90)
    if stats["train"][0] == 0:
        print("\nKeine Trainingsbilder. Zuerst: python tools/prelabel.py, in LabelImg korrigieren,\n"
              "Planname in data/geprueft.txt eintragen.")
        return
    if args.zip:
        print(f"\nZIP für Google Colab: {make_zip()}")
        print("Weiter im Notebook tools/yolo_train_colab.ipynb (ZIP hochladen, Training starten,\n"
              "best.pt herunterladen und als models/stairs_yolo.pt speichern).")
        return
    if args.build_only:
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nultralytics fehlt. Einmalig: python -m pip install ultralytics")
        return

    cfg = yd.yolo_settings(settings)
    model = YOLO(args.model)
    kwargs = dict(data=str(yaml), epochs=args.epochs, imgsz=int(cfg["tile"]), batch=args.batch,
                  project=str(RUNS_DIR), name=f"train_{datetime.now():%Y-%m-%d_%H-%M}",
                  patience=30, workers=0, fliplr=0.5, flipud=0.5, degrees=0.0,
                  hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, mosaic=1.0, plots=True)
    if args.device:
        kwargs["device"] = args.device
    print(f"\nTraining startet ({args.epochs} Epochen, Startmodell {args.model}) ...")
    model.train(**kwargs)
    best = Path(model.trainer.best)
    if not best.exists():
        print("Training beendet, aber best.pt wurde nicht gefunden.")
        return
    target = install_model(best)
    print(f"\nFertig. Modell: {target}")
    print(f"Lernkurven und Beispiele: {best.parent.parent}  (results.png, val_batch0_pred.jpg)")
    print("Jetzt im Fenster Methode 'YOLO' oder 'Beide vergleichen' wählen.")


if __name__ == "__main__":
    main()
