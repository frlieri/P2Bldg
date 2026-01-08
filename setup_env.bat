@echo off
setlocal

rem Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
  echo Python not found. Make sure Python is in PATH.
  exit /b 1
)

rem Create venv if it does not exist
if not exist "venv\Scripts\activate" (
  echo Creating virtual environment in `venv`...
  python -m venv venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
  )
) else (
  echo Virtual environment `venv` already exists.
)

rem Activate venv (use call so activation affects current process)
call "venv\Scripts\activate"

rem Upgrade pip/setuptools/wheel and install dependencies
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo Pip upgrade failed.
  exit /b 1
)

if exist "requirements.txt" (
  echo Installing dependencies from `requirements.txt`...
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
  )
) else (
  echo `requirements.txt` not found.
)

echo Done.
pause
endlocal
exit /b 0
