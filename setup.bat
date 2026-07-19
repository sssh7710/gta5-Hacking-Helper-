@echo off
setlocal
cd /d "%~dp0"
py -3.13 -m venv .venv
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Installation completed. Run run.bat.
exit /b 0
:error
echo Installation failed. Review the message above.
exit /b 1
