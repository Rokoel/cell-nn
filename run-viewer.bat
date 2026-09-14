@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Set up the viewer first: py -3.12 -m venv .venv
    echo Then run: .venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0run_viewer.py" %*
if errorlevel 1 pause
