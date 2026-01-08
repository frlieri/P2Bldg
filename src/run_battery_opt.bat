@echo off
rem Run this file from `src` folder.

rem directory of this script (ends with backslash)
set "SCRIPT_DIR=%~dp0"

rem compute project root (parent of script dir) robustly
pushd "%SCRIPT_DIR%.."
set "PROJECT_DIR=%CD%"
popd

rem activate venv located in project root
if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
  call "%PROJECT_DIR%\venv\Scripts\activate.bat"
) else (
  echo Virtual environment not found at `%PROJECT_DIR%\venv\Scripts\activate.bat`
  pause
  exit /b 1
)

rem prepend project root to PYTHONPATH so `src` package is importable
set "PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%"

rem run the script (forwards any CLI args)
python "%SCRIPT_DIR%run_battery_opt.py" %*

rem keep the cmd window open
pause
