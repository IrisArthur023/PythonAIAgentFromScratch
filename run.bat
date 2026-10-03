@echo off
setlocal
title Pixel
cd /d "%~dp0"

set "VENV=.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "FRONT=%~dp0..\pixel"

if not exist "%PYTHON%" (
    echo Creating a private Python environment for this project...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv "%VENV%"
    ) else (
        py -3 -m venv "%VENV%"
    )
    if errorlevel 1 goto :error

    echo Installing backend dependencies. This may take a few minutes the first time...
    "%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto :error
)

if not exist ".env" (
    echo.
    echo Missing .env file. Copy sample.env to .env and add your API keys, or use the Access tab after start.
    goto :error
)

if not exist "%FRONT%\node_modules" (
    echo Installing frontend dependencies...
    pushd "%FRONT%"
    npm install
    if errorlevel 1 (
        popd
        goto :error
    )
    popd
)

echo Starting Pixel API at http://127.0.0.1:8001
start "Pixel API" /min cmd /c "\"%PYTHON%\" main.py"
timeout /t 2 /nobreak >nul

echo Opening Pixel at http://localhost:5174
start "Open Pixel" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:5174'"
pushd "%FRONT%"
npm run dev
popd

echo.
echo Pixel interface stopped. Close the Pixel API window to stop the backend.
pause
exit /b 0

:error
echo.
echo The application could not start. Review the message above and try again.
pause
exit /b 1
