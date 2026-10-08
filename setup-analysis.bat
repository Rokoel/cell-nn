@echo off
py -3.12 -m venv "%~dp0.venv"
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%~dp0.venv\Scripts\python.exe" -m pip install -e "%~dp0.[neural]"
if errorlevel 1 goto failed
call "%~dp0setup-spots.bat"
if errorlevel 1 goto failed
echo Setup complete. Start run-analysis.bat.
pause
exit /b 0
:failed
echo Setup failed. Check that 64-bit Python 3.12 is installed.
pause
exit /b 1
