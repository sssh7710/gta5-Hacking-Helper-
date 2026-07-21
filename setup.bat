@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
py -3.13 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.13"
if not defined PYTHON_CMD (
  py -3.12 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=py -3.12"
)
if not defined PYTHON_CMD (
  py -3.11 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=py -3.11"
)
if not defined PYTHON_CMD (
  python -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD goto :python_missing

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 goto :python_version
%PYTHON_CMD% -c "import tkinter"
if errorlevel 1 goto :tkinter_missing

echo Using %PYTHON_CMD%
set "CREATE_VENV=1"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if not errorlevel 1 set "CREATE_VENV=0"
)
if "%CREATE_VENV%"=="1" (
  echo Creating virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo Reusing existing virtual environment.
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Installation completed. Run run.bat.
exit /b 0

:python_missing
echo.
echo [ERROR] Python was not found.
echo Install Python 3.13 from python.org and enable "Add python.exe to PATH".
pause
exit /b 1

:python_version
echo.
echo [ERROR] Python 3.10 or newer is required. Python 3.13 is recommended.
pause
exit /b 1

:tkinter_missing
echo.
echo [ERROR] This Python installation does not include tkinter.
echo Reinstall Python from python.org with Tcl/Tk support enabled.
pause
exit /b 1

:error
echo.
echo [ERROR] Installation failed. Review the message above.
echo If .venv was copied from another PC, delete only the .venv folder and run setup.bat again.
pause
exit /b 1
