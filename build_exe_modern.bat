@echo off
REM ==========================================================================
REM  Baut StairwayToHeavenModern.exe (modernes Fenster, Windows, ohne Python)
REM  Aufruf im VS-Code-Terminal im Projektordner:   .\build_exe_modern.bat
REM  Ergebnis: Ordner dist\  mit  StairwayToHeavenModern.exe, plans\, output\, settings.json
REM  Hinweis: Die exe enthaelt nur die Regel-Erkennung. YOLO (PyTorch, > 1 GB) wird bewusst
REM  NICHT eingepackt - YOLO laeuft ueber Python: python app_modern.py
REM ==========================================================================
cd /d "%~dp0"
echo [1/3] Pakete installieren ...
python -m pip install --upgrade pyinstaller customtkinter opencv-python numpy pillow
if errorlevel 1 goto fehler

echo [2/3] exe bauen (dauert 1-3 Minuten) ...
python -m PyInstaller --noconfirm --clean --onefile --windowed --name StairwayToHeavenModern --collect-data customtkinter --hidden-import PIL._tkinter_finder --hidden-import PIL.ImageTk --exclude-module torch --exclude-module torchvision --exclude-module ultralytics --paths src app_modern.py
if errorlevel 1 goto fehler

echo [3/3] Ordner neben der exe anlegen ...
if not exist dist\plans mkdir dist\plans
if not exist dist\output mkdir dist\output
copy /Y settings.json dist\settings.json >nul

echo.
echo Fertig:  dist\StairwayToHeavenModern.exe
echo Plaene (TIF) nach dist\plans kopieren und die exe per Doppelklick starten.
echo Zum Weitergeben den ganzen Ordner dist\ kopieren oder zippen.
goto ende

:fehler
echo.
echo Build fehlgeschlagen - bitte die Meldungen oben pruefen.

:ende
pause
