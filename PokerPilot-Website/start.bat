@echo off
rem Double-click to launch PokerPilot and open it in your browser.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :error
)

rem If PokerPilot is already running, just open it.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/ -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    start "" http://127.0.0.1:5000
    exit /b 0
)

start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:5000"
echo PokerPilot is running at http://127.0.0.1:5000
echo Close this window to stop it.
".venv\Scripts\python.exe" app.py
goto :eof

:error
echo Setup failed. Make sure Python is installed.
pause
