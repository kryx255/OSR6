@echo off
setlocal

set PROJECT_DIR=%~dp0
set PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe
set PYTHONW_EXE=%PROJECT_DIR%.venv\Scripts\pythonw.exe

pushd "%PROJECT_DIR%" >nul
if errorlevel 1 (
    echo Failed to enter the project directory.
    endlocal & exit /b 1
)

if exist "%PYTHON_EXE%" (
    set OSRGEN_PYTHON_EXE=%PYTHON_EXE%
    if exist "%PYTHONW_EXE%" (
        start "" "%PYTHONW_EXE%" -m osrgen.gui
    ) else (
        start "" "%PYTHON_EXE%" -m osrgen.gui
    )
) else (
    set OSRGEN_PYTHON_EXE=python
    start "" pythonw -m osrgen.gui
)

set EXIT_CODE=%ERRORLEVEL%
popd
endlocal & exit /b %EXIT_CODE%
