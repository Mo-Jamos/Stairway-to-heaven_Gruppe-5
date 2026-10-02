"""
Stairway to Heaven - Start ohne Fenster (für VS Code / Terminal)

    python main.py                 -> alle Pläne in plans/ verarbeiten (Batch)
    python main.py S_133_005       -> nur diesen Plan (Name ohne Endung reicht)
    python main.py --method yolo   -> mit YOLO statt Regeln ("rules", "yolo" oder "both")
    python main.py --method both   -> beide Methoden, danach: python src/evaluate.py
    python app_modern.py           -> Programm MIT Fenster starten

Ergebnisse: output/<Datum_Uhrzeit>_.../ (jeder Lauf in einem eigenen Ordner)
Einstellungen: settings.json (z. B. confidence_threshold)
Logik: src/stairway_app.py (Ein-/Ausgabe) und src/stair_detector.py (Treppenerkennung)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import stairway_app  # noqa: E402

if __name__ == "__main__":
    stairway_app.cli_main(sys.argv[1:])
