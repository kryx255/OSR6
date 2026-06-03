@echo off
setlocal

set PYTHON_EXE=%~dp0.venv\Scripts\python.exe

pushd "%~dp0" >nul
if errorlevel 1 (
    echo Failed to enter the project directory.
    endlocal & exit /b 1
)

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m osrgen %*
) else (
    python -m osrgen %*
)

set EXIT_CODE=%ERRORLEVEL%
popd
endlocal & exit /b %EXIT_CODE%
