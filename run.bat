@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment was not found. Starting setup...
  call setup.bat
  if errorlevel 1 exit /b 1
)

echo Starting GTA hacking helper...
echo Keep this window open while the helper is running.
".venv\Scripts\python.exe" -B app.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
  echo.
  echo [ERROR] The helper stopped with exit code %APP_EXIT%.
  echo Copy the error message above when asking for help.
  pause
)
exit /b %APP_EXIT%
