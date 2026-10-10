@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Run setup-analysis.bat first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0run_annotator.py" %*
if errorlevel 1 pause
