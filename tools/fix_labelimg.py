"""
Repariert den bekannten Fehler von LabelImg 1.8.6 mit Python >= 3.10:
    TypeError: setValue(self, val: int): argument 1 has unexpected type 'float'
Qt erwartet ganze Zahlen, LabelImg übergibt Kommazahlen. Dieses Skript ergänzt int(...).

Aufruf (einmalig, im Terminal im Ordner stairway_prototype):
    python tools/fix_labelimg.py
"""
import importlib.util
from pathlib import Path

FIXES = {
    ("labelImg", "labelImg.py"): [
        ("bar.setValue(bar.value() + bar.singleStep() * units)",
         "bar.setValue(int(bar.value() + bar.singleStep() * units))"),
        ("self.zoom_widget.setValue(value)", "self.zoom_widget.setValue(int(value))"),
        ("h_bar.setValue(new_h_bar_value)", "h_bar.setValue(int(new_h_bar_value))"),
        ("v_bar.setValue(new_v_bar_value)", "v_bar.setValue(int(new_v_bar_value))"),
    ],
    ("libs", "canvas.py"): [
        ("p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
         "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))"),
        ("p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
         "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))"),
        ("p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
         "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))"),
    ],
    ("libs", "shape.py"): [
        ("painter.drawText(min_x, min_y, self.label)",
         "painter.drawText(int(min_x), int(min_y), self.label)"),
    ],
}

for (package, filename), replacements in FIXES.items():
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        print(f"Paket '{package}' nicht gefunden - ist labelImg installiert?")
        continue
    path = Path(list(spec.submodule_search_locations)[0]) / filename
    text = path.read_text(encoding="utf-8")
    changed = 0
    for old, new in replacements:
        if new in text:
            continue                     # schon repariert
        if old in text:
            text = text.replace(old, new)
            changed += 1
        else:
            print(f"  Hinweis: Stelle nicht gefunden in {path.name}: {old[:50]}...")
    if changed:
        path.write_text(text, encoding="utf-8")
    print(f"{path}: {changed} Stelle(n) repariert")

print("Fertig. LabelImg neu starten.")
