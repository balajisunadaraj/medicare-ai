@echo off
REM Launch the Flask app using the project virtual environment.
set VENV=%~dp0.venv\Scripts\python.exe
if not exist "%VENV%" (
    echo Virtual environment not found at %~dp0.venv\Scripts\python.exe
    echo Create it with: python -m venv .venv
    exit /b 1
)
"%VENV%" "%~dp0app.py"
