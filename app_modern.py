"""
Stairway to Heaven - modernes Programmfenster (CustomTkinter)

Start:  python app_modern.py          (vorher einmalig: python -m pip install customtkinter)
        python app_modern.py --batch  -> ohne Fenster alle Pläne verarbeiten

Funktionen:
  - Hell / Dunkel / System-Design
  - Einzelplan (Dropdown mit Filter beim Tippen) und Batch mit Fortschritt + Abbrechen
  - Vorschau des letzten Ergebnisses direkt im Fenster
  - Knöpfe für Ergebnisbild, HTML-Bericht (report.html) und Ergebnisordner
  - Erkennungsmethode umschalten: Regeln, YOLO oder beide zum Vergleichen

Die Logik steckt in src/stairway_app.py (gleich wie bei main.py).
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import stairway_app as core  # noqa: E402
import yolo_detector         # noqa: E402

TITLE = "Stairway to Heaven"
ACCENT = ("#c8102e", "#e0404f")            # (hell, dunkel)
ACCENT_HOVER = ("#a00d25", "#c43341")
PREVIEW_SIZE = (520, 330)
METHOD_LABELS = {"rules": "Rules", "yolo": "YOLO", "both": "Compare both"}


def make_logo(size=44):
    """Logo: weiße Treppe auf rotem, abgerundetem Quadrat (ohne externe Bilddatei)."""
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=10 * scale, fill=(200, 16, 46, 255))
    step, x, y = s // 6, s // 5, s - s // 4
    points = [(x, y)]
    for _ in range(3):
        points += [(x, y - step), (x + step, y - step)]
        x, y = x + step, y - step
    points.append((x + step // 2, y))
    d.line(points, fill="white", width=int(2.6 * scale), joint="curve")
    return img.resize((size, size), Image.LANCZOS)


def open_path(path):
    """Datei oder Ordner mit dem Standardprogramm öffnen."""
    if sys.platform.startswith("win"):
        os.startfile(str(path))                      # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.settings = core.load_settings()
        self.events = queue.Queue()
        self.cancel_requested = threading.Event()
        self.busy = False
        self.last_overlay = None
        self.last_report = None
        self.last_run_dir = None
        self.preview_image = None

        core.PLANS_DIR.mkdir(exist_ok=True)
        core.OUTPUT_DIR.mkdir(exist_ok=True)
        self.title(f"{TITLE} – Staircase Detection {core.PROGRAM_VERSION}")
        self.geometry("1060x720")
        self.minsize(900, 660)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.build_header()
        self.build_controls()
        self.build_preview()
        self.build_footer()
        self.refresh_plans()
        self.after(100, self.poll_events)

    # ------------------------------------------------------------------ Aufbau
    def build_header(self):
        head = ctk.CTkFrame(self, corner_radius=0, fg_color=("#ffffff", "#1a202a"), height=72)
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        head.grid_columnconfigure(1, weight=1)
        self.logo_image = ctk.CTkImage(light_image=make_logo(), dark_image=make_logo(), size=(44, 44))
        logo = ctk.CTkLabel(head, image=self.logo_image, text="")
        logo.grid(row=0, column=0, rowspan=2, padx=(20, 12), pady=14)
        ctk.CTkLabel(head, text=TITLE, font=ctk.CTkFont(size=20, weight="bold"),
                     anchor="w").grid(row=0, column=1, sticky="sw", pady=(14, 0))
        ctk.CTkLabel(head, text="Staircase detection in BVG floor plans · runs locally, no internet",
                     text_color=("gray40", "gray65"), anchor="w").grid(row=1, column=1, sticky="nw")
        self.mode = ctk.CTkSegmentedButton(head, values=["System", "Light", "Dark"],
                                           command=self.change_mode, selected_color=ACCENT,
                                           selected_hover_color=ACCENT_HOVER)
        self.mode.set("System")
        self.mode.grid(row=0, column=2, rowspan=2, padx=20)

    def card(self, parent, title, row):
        frame = ctk.CTkFrame(parent, corner_radius=12)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=15, weight="bold"),
                     anchor="w").grid(row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(12, 4))
        return frame

    def build_controls(self):
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsw", padx=(20, 10), pady=20)
        left.grid_columnconfigure(0, weight=1)

        method = self.card(left, "Detection method", 0)
        self.method_btn = ctk.CTkSegmentedButton(method, values=list(METHOD_LABELS.values()),
                                                 command=self.change_method, selected_color=ACCENT,
                                                 selected_hover_color=ACCENT_HOVER)
        self.method_btn.grid(row=1, column=0, sticky="w", padx=16, pady=(4, 4))
        self.method_info = ctk.CTkLabel(method, text="", text_color=("gray40", "gray65"), anchor="w",
                                        justify="left", wraplength=330)
        self.method_info.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 12))
        start = self.settings.get("method", "rules")
        if start != "rules" and not yolo_detector.status(self.settings)[0]:
            start = "rules"
        self.settings["method"] = start
        self.method_btn.set(METHOD_LABELS[start])
        self.update_method_info()

        single = self.card(left, "1   Single plan", 1)
        ctk.CTkLabel(single, text="Select a plan or type its name (no file extension needed)",
                     text_color=("gray40", "gray65"), anchor="w").grid(row=1, column=0, columnspan=2,
                                                                      sticky="w", padx=16)
        self.combo = ctk.CTkComboBox(single, values=[], width=280, command=lambda v: None)
        self.combo.set("")
        self.combo.grid(row=2, column=0, sticky="ew", padx=(16, 8), pady=(6, 14))
        self.combo.bind("<KeyRelease>", self.on_type)
        self.combo.bind("<Return>", lambda e: self.on_open())
        self.open_btn = ctk.CTkButton(single, text="Open", width=90, command=self.on_open,
                                      fg_color=ACCENT, hover_color=ACCENT_HOVER)
        self.open_btn.grid(row=2, column=1, padx=(0, 16), pady=(6, 14))

        batch = self.card(left, "2   Batch processing", 2)
        ctk.CTkLabel(batch, text="Process all plans in the plans/ folder one after another",
                     text_color=("gray40", "gray65"), anchor="w").grid(row=1, column=0, columnspan=2,
                                                                      sticky="w", padx=16)
        buttons = ctk.CTkFrame(batch, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 6))
        self.batch_btn = ctk.CTkButton(buttons, text="Process all plans", command=self.on_batch,
                                       fg_color=ACCENT, hover_color=ACCENT_HOVER)
        self.batch_btn.pack(side="left")
        self.cancel_btn = ctk.CTkButton(buttons, text="Cancel", width=90, command=self.on_cancel,
                                        state="disabled", fg_color="transparent", border_width=1,
                                        text_color=("gray20", "gray85"))
        self.cancel_btn.pack(side="left", padx=8)
        self.progress = ctk.CTkProgressBar(batch, progress_color=ACCENT)
        self.progress.set(0)
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(6, 4))
        self.status = ctk.CTkLabel(batch, text="Ready.", anchor="w", wraplength=330, justify="left")
        self.status.grid(row=4, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 14))

    def build_preview(self):
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=20)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(right, text="Latest result", font=ctk.CTkFont(size=15, weight="bold"),
                     anchor="w").grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        self.preview = ctk.CTkLabel(right, text="No result yet.\nOpen a plan or start a batch run.",
                                    text_color=("gray40", "gray65"), fg_color=("#f2f4f8", "#232a35"),
                                    corner_radius=10, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)
        self.summary = ctk.CTkLabel(right, text="", anchor="w", justify="left")
        self.summary.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 4))
        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="w", padx=16, pady=(4, 14))
        self.img_btn = ctk.CTkButton(actions, text="Open result image", state="disabled",
                                     command=lambda: self.open_or_warn(self.last_overlay))
        self.img_btn.pack(side="left")
        self.report_btn = ctk.CTkButton(actions, text="Open report (HTML)", state="disabled",
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                        command=lambda: self.open_or_warn(self.last_report))
        self.report_btn.pack(side="left", padx=8)
        self.folder_btn = ctk.CTkButton(actions, text="Result folder", state="disabled",
                                        command=lambda: self.open_or_warn(self.last_run_dir))
        self.folder_btn.pack(side="left")
        for b in (self.img_btn, self.report_btn, self.folder_btn):
            self.enable(b, False)

    def build_footer(self):
        foot = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        foot.grid(row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 14))
        ctk.CTkButton(foot, text="Open output folder", width=160, fg_color="transparent", border_width=1,
                      text_color=("gray20", "gray85"),
                      command=lambda: self.open_or_warn(core.OUTPUT_DIR)).pack(side="left")
        ctk.CTkButton(foot, text="Refresh plan list", width=170, fg_color="transparent",
                      border_width=1, text_color=("gray20", "gray85"),
                      command=self.refresh_plans).pack(side="left", padx=8)
        self.count = ctk.CTkLabel(foot, text="", text_color=("gray40", "gray65"))
        self.count.pack(side="right")

    # ------------------------------------------------------------------ Methode
    def change_method(self, label):
        method = {v: k for k, v in METHOD_LABELS.items()}[label]
        if method != "rules":
            ok, message = yolo_detector.status(self.settings)
            if not ok:
                messagebox.showinfo(TITLE, message)
                self.method_btn.set(METHOD_LABELS[self.settings.get("method", "rules")])
                return
        self.settings["method"] = method
        self.update_method_info()

    def update_method_info(self):
        method = self.settings.get("method", "rules")
        ok, message = yolo_detector.status(self.settings)
        texts = {
            "rules": "Rules: fixed features (tread lines, walk line). No training needed.",
            "yolo": "YOLO: learned model. " + (message if ok else ""),
            "both": "Both: red boxes = rules, blue boxes = YOLO. Then run evaluate.py.",
        }
        if ok or method != "rules":
            extra = ""
        elif "not installed" in message:
            extra = "\nYOLO not installed yet: python -m pip install ultralytics"
        else:
            extra = "\nNo YOLO model yet: train first (tools/yolo_train.py)"
        self.method_info.configure(text=texts[method] + extra)

    # ------------------------------------------------------------------ Hilfsfunktionen
    def change_mode(self, value):
        ctk.set_appearance_mode({"System": "system", "Light": "light", "Dark": "dark"}[value])

    def refresh_plans(self):
        names = [p.name for p in core.list_plans()]
        self.combo.configure(values=names)
        self.count.configure(text=f"{len(names)} plans in {core.PLANS_DIR}  ·  "
                                  f"Version {core.PROGRAM_VERSION}")

    def on_type(self, event):
        if event.keysym in ("Return", "Up", "Down", "Escape"):
            return
        self.combo.configure(values=core.filter_plans(self.combo.get()))

    def open_or_warn(self, path):
        try:
            open_path(path)
            return True
        except Exception:                            # noqa: BLE001
            messagebox.showerror(TITLE, "The result image could not be opened.")
            return False

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for w in (self.open_btn, self.batch_btn, self.combo, self.method_btn):
            w.configure(state=state)
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    @staticmethod
    def enable(button, on, primary=False):
        """Knopf aktivieren/deaktivieren - deaktiviert sieht er grau aus."""
        if on:
            button.configure(state="normal", fg_color=ACCENT if primary else ("#3b8ed0", "#1f6aa5"))
        else:
            button.configure(state="disabled", fg_color=("gray75", "gray30"))

    def show_result(self, overlay, report, run_dir, text):
        self.last_overlay, self.last_report, self.last_run_dir = overlay, report, run_dir
        self.summary.configure(text=text)
        self.enable(self.img_btn, overlay)
        self.enable(self.report_btn, report, primary=True)
        self.enable(self.folder_btn, run_dir)
        thumb = Path(str(overlay).replace("_overlay.png", "_thumb.jpg")) if overlay else None
        source = thumb if thumb and thumb.exists() else overlay
        if source:
            try:
                img = Image.open(source)
                img.thumbnail(PREVIEW_SIZE)
                self.preview_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self.preview.configure(image=self.preview_image, text="")
                return
            except Exception:                        # noqa: BLE001
                pass
        self.preview.configure(image=None, text="No preview available.")

    def poll_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                getattr(self, f"handle_{kind}")(data)
        except queue.Empty:
            pass
        self.after(100, self.poll_events)

    # ------------------------------------------------------------------ Einzelplan
    def on_open(self):
        if self.busy:
            return
        name = self.combo.get().strip()
        if not name:
            messagebox.showwarning(TITLE, "Please enter a plan name.")
            return
        plan = core.find_plan(name)
        if plan is None:
            messagebox.showerror(TITLE, f"The plan '{name}' is not available in the folder. "
                                        f"Please add the plan or choose a different one.")
            return
        self.set_busy(True)
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        self.status.configure(text=f"Processing {plan.name} …")
        threading.Thread(target=self.single_worker, args=(plan,), daemon=True).start()

    def single_worker(self, plan):
        try:
            self.events.put(("single_done", core.run_single(plan, self.settings)))
        except Exception as e:                       # noqa: BLE001
            self.events.put(("single_failed", (plan, e)))

    def stop_progress(self, value=0):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(value)

    def handle_single_done(self, info):
        self.stop_progress()
        self.set_busy(False)
        dets = info["result"]["detections"]
        review = sum(d["needs_review"] for d in dets)
        overlay = info["overlay_paths"][0] if info["overlay_paths"] else None
        if info["result"].get("method") == "both":
            by = info["result"]["stairs_by_method"]
            found = f"Rules {by.get('rules', 0)}, YOLO {by.get('yolo', 0)} staircase flights"
        else:
            found = f"{len(dets)} staircase flight{'' if len(dets) == 1 else 's'} detected"
        self.show_result(overlay, info.get("report_path"), info["run_dir"],
                         f"{info['plan_name']}:  {found}"
                         f"{f', {review} to review' if review else ''}")
        self.status.configure(text=f"Done: {info['run_dir'].name}")
        if info["save_error"]:
            messagebox.showerror(TITLE, "The result files could not be saved.")
            return
        if info["image_error"] or not overlay:
            messagebox.showerror(TITLE, "The result image could not be opened.")
            return
        try:
            open_path(overlay)
        except Exception:                            # noqa: BLE001
            messagebox.showerror(TITLE, "The result image could not be opened.")
            return
        messagebox.showinfo(TITLE, "The result was saved successfully.")

    def handle_single_failed(self, data):
        plan, error = data
        self.stop_progress()
        self.set_busy(False)
        self.status.configure(text=f"Error: {plan.name}")
        messagebox.showerror(TITLE, f"The plan '{plan.name}' could not be processed.\n\n{error}")

    # ------------------------------------------------------------------ Batch
    def on_batch(self):
        if self.busy:
            return
        plans = core.list_plans()
        if not plans:
            messagebox.showwarning(TITLE, f"No plans found. Please add TIF files to:\n{core.PLANS_DIR}")
            return
        self.cancel_requested.clear()
        self.set_busy(True)
        self.progress.set(0)
        threading.Thread(target=self.batch_worker, args=(plans,), daemon=True).start()

    def batch_worker(self, plans):
        def progress(i, n, name):
            self.events.put(("progress", (i, n, name)))
        try:
            summary = core.run_batch(plans, self.settings, progress=progress,
                                     cancelled=self.cancel_requested.is_set)
            self.events.put(("batch_done", summary))
        except Exception as e:                       # noqa: BLE001
            self.events.put(("batch_failed", e))

    def on_cancel(self):
        self.cancel_requested.set()
        self.status.configure(text="Stopping after the current plan …")

    def handle_progress(self, data):
        i, n, name = data
        self.progress.set((i - 1) / n)
        self.status.configure(text=f"Processing plan {i} of {n}: {name}")

    def handle_batch_done(self, s):
        self.progress.set(1)
        self.set_busy(False)
        self.status.configure(text=f"Done: {s['run_dir'].name}")
        overlays = sorted(Path(s["run_dir"]).rglob("*_overlay.png"))
        self.show_result(overlays[0] if overlays else None, s.get("report_path"), s["run_dir"],
                         f"Batch: {s['ok']} of {s['total']} plans successful, {s['failed']} failed. "
                         f"Details in the report.")
        text = (f"Batch processing finished: {s['ok']} of {s['total']} plans processed "
                f"successfully, {s['failed']} failed. See processing_log.txt for details.")
        if s["cancelled"]:
            text = "Batch processing was cancelled.\n\n" + text
        messagebox.showinfo(TITLE, text)

    def handle_batch_failed(self, error):
        self.set_busy(False)
        messagebox.showerror(TITLE, f"Batch processing failed:\n{error}")


def main():
    if "--batch" in sys.argv:                        # ohne Fenster
        core.cli_main([a for a in sys.argv[1:] if a != "--batch"])
        return
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    ModernApp().mainloop()


if __name__ == "__main__":
    main()
