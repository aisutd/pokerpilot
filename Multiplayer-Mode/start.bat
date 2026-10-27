@echo off
rem Double-click to launch PokerPilot Online and open it in your browser.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :error
)

rem Table chat: the chat box UI (client) and its WebSocket relay (chat-api, port 3000).
where npm >nul 2>nul || goto :node_error
if not exist "client\node_modules" (
    echo Installing chat UI dependencies...
    call npm ci --prefix client || goto :chat_error
)
if not exist "client\dist\index.html" (
    echo Building chat UI...
    call npm run build --prefix client || goto :chat_error
)
if not exist "chat-api\node_modules" (
    echo Installing chat server dependencies...
    call npm ci --prefix chat-api || goto :chat_error
)
netstat -ano | findstr /R /C:":3000 .*LISTENING" >nul
if errorlevel 1 start "PokerPilot Chat API" /min cmd /c "cd /d ""%~dp0chat-api"" && npm start"

rem If PokerPilot Online is already running, just open it.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5001/ -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    start "" http://127.0.0.1:5001
    exit /b 0
)

start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:5001"
echo PokerPilot Online is running at http://127.0.0.1:5001
echo The chat server runs in a separate minimized window.
echo Close this window to stop it.
".venv\Scripts\python.exe" app.py
goto :eof

:error
echo Setup failed. Make sure Python is installed.
pause
exit /b 1

:node_error
echo Setup failed. Install Node.js 20.19 or newer to run the table chat.
pause
exit /b 1

:chat_error
echo Setup failed while installing or building the table chat.
pause
exit /b 1
