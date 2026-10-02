"""
Stairway to Heaven - Evaluation (für Nerijus)

Vergleicht die Ergebnisse eines Laufs (output/<Datum>_batch_.../<plan>/<plan>_result.json)
mit den Annotationen (data/annotations/<plan>.txt im YOLO-Format von LabelImg).
Ohne --dir wird automatisch der NEUESTE Batch-Lauf in output/ ausgewertet.

YOLO-Format, eine Zeile pro Box:  <klasse> <x_mitte> <y_mitte> <breite> <höhe>
(alle Werte relativ zur Bildgröße, 0 bis 1)
Klassen: 0 = stair_plan, 1 = stair_section, 2 = hard_negative
Standard: Klasse 0 (Draufsicht). Mit --class 1 wird die Seitenansicht ausgewertet.
hard_negative-Boxen werden ignoriert.

Aufruf:
    python src/evaluate.py                        -> alle Pläne mit JSON + Annotation
    python src/evaluate.py data/split_val.txt     -> nur Pläne aus der Liste
    python src/evaluate.py data/split_test.txt    -> Endauswertung (Freitag!)
    python src/evaluate.py data/split_val.txt --class 1   -> Seitenansichten
    python src/evaluate.py data/split_val.txt --dir output/2026-10-01_09-15_batch_4-plans
    python src/evaluate.py data/split_val.txt --dir outputs/v01   -> Ergebnisse von v0.1
    python src/evaluate.py data/split_val.txt --method yolo       -> nur YOLO-Boxen

Regeln oder YOLO? Einen Lauf mit Methode "Beide vergleichen" machen (Fenster) bzw.
    python main.py --method both
und danach evaluate.py aufrufen: Precision/Recall/F1 werden für JEDE Methode getrennt
berechnet und nebeneinander ausgegeben.

ACHTUNG: Pläne, deren Annotation aus Regel-Vorschlägen (tools/prelabel.py) entstanden ist,
bevorzugen die Regeln leicht. Für die Endauswertung (split_test) daher ohne Vorschläge annotieren.

Die Listendateien enthalten einen Dateinamen ohne Endung pro Zeile, z. B. G_90_a
Nur Python-Standardbibliothek, keine Installation nötig.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = None                     # None = neuester Batch-Lauf in output/
METHOD = None                      # None = jede Methode im Lauf getrennt auswerten
METHOD_NAMES = {"rules": "Regeln", "yolo": "YOLO"}
ANN_DIR = PROJECT_DIR / "data" / "annotations"
IOU_THRESHOLD = 0.5
EVAL_CLASS = 0
CLASS_NAMES = ["stair_plan", "stair_section", "hard_negative"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_annotations(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != EVAL_CLASS:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:])
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def newest_batch_run():
    runs = sorted((PROJECT_DIR / "output").glob("*_batch_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else PROJECT_DIR / "outputs"


def load_predictions(folder):
    """
    Liest Ergebnisse im neuen Format (<plan>_result.json, bbox = [x, y, Breite, Höhe])
    und im alten Format von v0.1/v0.2 (<plan>.json, bbox = [x1, y1, x2, y2]).
    Rückgabe: {plan_name: {"size": (w, h), "detections": [{"bbox", "class", "score"}]}}
    """
    plans = {}
    new_files = sorted(folder.rglob("*_result.json"))
    for f in new_files:
        r = json.loads(f.read_text(encoding="utf-8"))
        dets = [{"bbox": [x, y, x + w, y + h], "class": d["view"], "score": d["confidence"],
                 "method": d.get("method", "rules")}
                for d in r["detections"] if d.get("page", 1) == 1
                for x, y, w, h in [d["bbox"]]]
        plans[r["plan_name"]] = {"size": r["image_size"], "detections": dets}
    if not new_files:
        for f in sorted(folder.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            dets = [{"bbox": d["bbox"], "class": d.get("class", "stair_plan"), "score": d["score"],
                     "method": "rules"} for d in r["detections"]]
            plans[f.stem] = {"size": r["image_size"], "detections": dets}
    return plans


def match(predictions, truths):
    """Greedy-Zuordnung nach Score. Jede Ground-Truth-Box darf nur einmal treffen."""
    predictions = sorted(predictions, key=lambda d: d["score"], reverse=True)
    used = set()
    tp = 0
    for p in predictions:
        best, best_iou = None, IOU_THRESHOLD
        for i, t in enumerate(truths):
            if i not in used and iou(p["bbox"], t) >= best_iou:
                best, best_iou = i, iou(p["bbox"], t)
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(predictions) - tp
    fn = len(truths) - tp
    return tp, fp, fn


def ratios(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(preds, names, method):
    rows, total = [], [0, 0, 0]
    for stem, pred in sorted(preds.items()):
        ann_file = ANN_DIR / f"{stem}.txt"
        if (names and stem not in names) or not ann_file.exists():
            continue
        width, height = pred["size"]
        truths = read_annotations(ann_file, width, height)
        predictions = [d for d in pred["detections"]
                       if d["class"] == CLASS_NAMES[EVAL_CLASS] and d["method"] == method]
        tp, fp, fn = match(predictions, truths)
        p, r, f = ratios(tp, fp, fn)
        rows.append([method, stem, len(truths), len(predictions), tp, fp, fn,
                     round(p, 3), round(r, 3), round(f, 3)])
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
    return rows, total


def main():
    global EVAL_CLASS, OUT_DIR, METHOD
    args = sys.argv[1:]
    if "--class" in args:
        i = args.index("--class")
        EVAL_CLASS = int(args[i + 1])
        del args[i:i + 2]
    if "--dir" in args:
        i = args.index("--dir")
        OUT_DIR = PROJECT_DIR / args[i + 1]
        del args[i:i + 2]
    if "--method" in args:
        i = args.index("--method")
        METHOD = args[i + 1]
        del args[i:i + 2]
    names = None
    if args:
        names = {l.strip() for l in Path(args[0]).read_text(encoding="utf-8").splitlines() if l.strip()}

    if OUT_DIR is None:
        OUT_DIR = newest_batch_run()
    print(f"Ausgewertet wird: {OUT_DIR}")
    preds = load_predictions(OUT_DIR)
    methods = [METHOD] if METHOD else sorted({d["method"] for p in preds.values() for d in p["detections"]}
                                              or {"rules"})
    all_rows, summary = [], []
    for method in methods:
        rows, total = evaluate(preds, names, method)
        if not rows:
            continue
        all_rows += rows
        print(f"\n=== {METHOD_NAMES.get(method, method)} ===")
        print(f'{"plan":<20}{"GT":>4}{"det":>5}{"TP":>4}{"FP":>4}{"FN":>4}{"P":>7}{"R":>7}{"F1":>7}')
        for r in rows:
            print(f"{r[1][:20]:<20}{r[2]:>4}{r[3]:>5}{r[4]:>4}{r[5]:>4}{r[6]:>4}{r[7]:>7.2f}{r[8]:>7.2f}{r[9]:>7.2f}")
        p, r, f = ratios(*total)
        print("-" * 62)
        print(f'{"GESAMT":<20}{"":>4}{"":>5}{total[0]:>4}{total[1]:>4}{total[2]:>4}{p:>7.2f}{r:>7.2f}{f:>7.2f}')
        summary.append([method, *total, round(p, 3), round(r, 3), round(f, 3)])

    if not all_rows:
        print("Keine passenden Paare aus Ergebnissen und data/annotations/*.txt gefunden.")
        return
    print(f"\n(Klasse {EVAL_CLASS} = {CLASS_NAMES[EVAL_CLASS]}, Treffer = IoU >= {IOU_THRESHOLD})")
    if len(summary) > 1:
        print("\nVERGLEICH          TP   FP   FN  Precision  Recall     F1")
        for s in summary:
            print(f"{METHOD_NAMES.get(s[0], s[0]):<16}{s[1]:>5}{s[2]:>5}{s[3]:>5}{s[4]:>11.2f}{s[5]:>8.2f}{s[6]:>7.2f}")

    header = ["method", "plan", "treppen_gt", "erkannt", "TP", "FP", "FN", "precision", "recall", "f1"]
    with open(OUT_DIR / "evaluation.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(all_rows)
        for s in summary:
            writer.writerow([s[0], "GESAMT", "", "", *s[1:]])
    print(f"Gespeichert: {OUT_DIR / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
