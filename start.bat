@echo off
echo Starting AI-Image-Suite...
echo.

REM Start Backend Server in new window
echo Starting Backend Server...
start "Backend Server" cmd /k "cd /d %~dp0 && .\myenv\Scripts\activate.bat && cd server && python app.py"

REM Wait for backend to start (increase if needed)
echo Waiting for backend server to start...
timeout /t 10 /nobreak >nul

REM Start Frontend in new window
echo Starting Frontend...
start "Frontend" cmd /k "cd /d %~dp0 && .\myenv\Scripts\activate.bat && npm start"

echo.
echo Both servers are starting...
echo Backend Server window and Frontend window should open shortly.
echo.
pause
