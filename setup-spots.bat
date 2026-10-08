@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv-spots
if errorlevel 1 exit /b 1
.venv-spots\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv-spots\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
.venv-spots\Scripts\python.exe -m pip install -r requirements-spots.txt
if errorlevel 1 exit /b 1
