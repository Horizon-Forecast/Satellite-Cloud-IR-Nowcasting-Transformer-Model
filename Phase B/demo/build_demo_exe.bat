@echo off
REM Build the standalone demo .exe (no torch). Run from project root or this folder.
cd /d "%~dp0"
echo Building HorizonForecastDemo.exe ...
"..\venv\Scripts\python.exe" -m PyInstaller --onefile --windowed --noconfirm ^
  --name HorizonForecastDemo ^
  --icon horizon_forecast.ico ^
  --add-data "assets;assets" ^
  --add-data "manifest.json;." ^
  app.py
echo.
echo Done. Output: dist\HorizonForecastDemo.exe
pause
