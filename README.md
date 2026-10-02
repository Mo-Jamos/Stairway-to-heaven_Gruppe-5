# Stairway to Heaven – Staircase Detection in BVG Floor Plans

Prototype of team **Stairway to Heaven (Group 5)** for the course *Informatik und Gesellschaft* at HTW Berlin, with BVG as practice partner. Version 0.4, October 2026.

The program finds staircases in scanned floor plans of U-Bahn stations. For every staircase flight it returns a bounding box, the view (top view or side view), the stair type, a confidence score and a review flag – as a marked image, JSON, CSV and an HTML report. It runs offline on a normal laptop.

> **No data in this repository.** BVG plans, annotations, example screenshots, results and the trained YOLO model are not published. Put your own plans into `plans/` (see *Usage*). The `.gitignore` keeps all of these files out of Git.

## Features

- **Two detection methods:** a rule-based detector (main method, no training data needed) and a YOLO11n model (optional, learned from annotated plans). Both can run on the same plan for comparison.
- **Top view** (`stair_plan`) and **side view** (`stair_section`); stair type straight, L-shaped, U-shaped or other.
- Stairs at any angle, broken tread lines, very large and multi-page TIFs (also 1-bit / G4).
- Window app (single plan or batch with progress bar), command line, and a Windows `.exe` (rules only).
- Every run gets its own output folder; earlier results are never overwritten. An error in one plan does not stop a batch.
- Evaluation script: precision, recall and F1 at IoU ≥ 0.5.

## Results (v0.4)

Measured on 4 hand-annotated plans (24 staircase flights), a detection counts if it overlaps the true box by IoU ≥ 0.5:

| Method | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Rules | 0.95 | 0.87 | 0.91 | 20 of 23 top-view flights found, 1 false alarm; side view 1 of 1 |
| YOLO11n | 0.57 | 0.34 | – | validation plan only, mAP50 0.40; trained on 3 plans |

These figures are optimistic: the rules were tuned on the same plans, and no independent test set exists yet (see *Limitations*).

## Installation

Python 3.11 is required.

```bash
python -m pip install -r requirements.txt
python -m pip install ultralytics   # optional, only for the YOLO method (about 1 GB with PyTorch)
```

## Usage

1. Copy your plans (TIF, PNG or JPG) into `plans/`.
2. Start the program:

```bash
python app_modern.py                    # window: choose a method, open one plan or process all plans
python main.py                          # all plans, without window
python main.py <plan name>              # one plan (case-insensitive, file extension optional)
python main.py --method yolo            # YOLO only   (needs models/stairs_yolo.pt)
python main.py --method both            # rules and YOLO side by side
```

3. Open the result: the window shows a preview and buttons for the result image, the HTML report and the result folder.

A Windows program without Python can be built with `build_exe_modern.bat` (output in `dist/`, rule-based method only).

## Output

Each run writes to `output/<date>_<time>_<plan or batch>/` (suffix `_yolo` or `_both` for the other methods):

| File | Content |
|---|---|
| `<plan>_overlay.png` | Plan with a numbered box per flight (rules red, YOLO blue; orange / light blue = needs review) |
| `<plan>_result.json` | All detections: box `[x, y, width, height]` in original pixels, view, stair type, confidence, review flag, method; plus program version and settings |
| `all_detections.csv` | One row per flight (semicolon + decimal comma for German Excel) |
| `summary_all_plans.csv` | Batch runs: one row per plan |
| `report.html` | Offline report with key figures, previews, zoomable overlays and a searchable table |
| `processing_log.txt` | Batch runs: course of the run and error details |

## Configuration (`settings.json`)

| Key | Default | Meaning |
|---|---|---|
| `confidence_threshold` | 0.7 | Rules: below this, `needs_review = true` |
| `csv_locale` | `de` | `de` = semicolon + decimal comma, `en` = comma + decimal point |
| `method` | `rules` | Start method: `rules`, `yolo` or `both` |
| `detector` | `{}` | Overrides for the rule thresholds in `CONFIG` (`src/stair_detector.py`) |
| `yolo.review_threshold` | 0.5 | YOLO: below this, `needs_review = true` |
| `yolo.min_confidence` | 0.25 | YOLO: weaker boxes are discarded |

## How it works

**Rules (`src/stair_detector.py`)**

1. Scale every page to 100 DPI, binarise with Sauvola's local threshold, remove small noise.
2. Find the main line directions with a Hough transform (stairs at any angle; 45°/135° hatching is skipped).
3. Group parallel, similarly long lines with regular spacing into flights (at least 5 treads).
4. Reject look-alikes: irregular spacing, tables and grids, text blocks, hatching, missing walk line.
5. Detect side views as regular zigzag step profiles; group flights into staircases and assign the stair type and a confidence score.

**YOLO (`src/yolo_detector.py`)** – YOLO11n (Ultralytics), fine-tuned on the team's annotations. Plans are processed in 1024 × 1024 px tiles with 256 px overlap; boxes cut at tile edges are merged. The trained model is not included – train your own as described below.

## Evaluation and training

Annotations are stored in YOLO format in `data/annotations/<plan>.txt` (classes in `data/classes.txt`: `stair_plan`, `stair_section`).

```bash
python main.py --method both                       # run both methods
python src/evaluate.py                             # precision / recall / F1 per plan and method
python src/evaluate.py data/split_test.txt         # only the plans in a list
python src/evaluate.py --class 1                   # side views

python tools/prelabel.py                           # box proposals from the rules, to correct in LabelImg
python tools/yolo_train.py --epochs 50             # train YOLO -> models/stairs_yolo.pt
python tools/check_examples.py                     # check example screenshots in data/examples/
```

Only plans listed in `data/geprueft.txt` are used for training; plans in `data/split_val.txt` are used for validation, plans in `data/split_test.txt` never for training. These three lists are not in the repository – create them yourself as plain text files with one plan name per line (e.g. `S_133_005`).


## Project structure

```text
app_modern.py            window app
main.py                  command line
settings.json            settings
requirements.txt         Python packages
build_exe_modern.bat     builds the Windows .exe
src/
  stair_detector.py      rule-based detection (all thresholds in CONFIG)
  yolo_detector.py       YOLO detection (tiling, merging)
  stairway_app.py        folders, JSON / CSV / overlay output, batch
  report.py              HTML report
  evaluate.py            precision, recall, F1
  examples.py            helpers for example screenshots
  stair_detector_v01.py  first version v0.1, kept for comparison
tools/                   prelabelling, YOLO training, example check, LabelImg helpers
plans/                   your plans (not in the repository)
data/                    annotations and examples (not in the repository)
output/                  results (not in the repository)
```


## Limitations

- Developed and evaluated on 4 plans only; the rules were tuned on the same plans, and the ground truth was created by correcting the program's own proposals.
- Missed: flights with fewer than 5 visible treads, without a clear walk line, very faint scans, heavy hatching, sloped side views; spiral stairs are not supported.
- Escalators are reported as normal staircases (no separate class).
- The confidence score is a rule score, not a calibrated probability.
- YOLO is limited by the very small training set.

## Author

<<<<<<< HEAD
Developed by **Mohammad AL Jamous** ([@Mo-Jamos](https://github.com/Mo-Jamos)) – design, implementation, YOLO training and evaluation.
=======
Developed by **Mohammad Al Jamous** ([@Mo-Jamos](https://github.com/Mo-Jamos)) – design, implementation, YOLO training and evaluation.
>>>>>>> 97fd19acff54fab12646319145b1b2d1f6ef4967
