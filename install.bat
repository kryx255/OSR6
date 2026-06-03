@echo off
setlocal

set PROJECT_DIR=%~dp0
set UV_EXE=

pushd "%PROJECT_DIR%" >nul
if errorlevel 1 (
    echo Failed to enter the project directory.
    endlocal & exit /b 1
)

if exist "%PROJECT_DIR%.venv\Scripts\uv.exe" (
    set UV_EXE=%PROJECT_DIR%.venv\Scripts\uv.exe
)

if not defined UV_EXE (
    where uv >nul 2>nul
    if not errorlevel 1 set UV_EXE=uv
)

if not defined UV_EXE (
    echo uv was not found in PATH.
    where winget >nul 2>nul
    if errorlevel 1 (
        echo Please install uv from https://docs.astral.sh/uv/getting-started/installation/
        echo Then run install.bat again.
        popd
        endlocal & exit /b 1
    )
    echo Installing uv with winget...
    winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo uv installation failed.
        popd
        endlocal & exit /b 1
    )
)

if not defined UV_EXE (
    where uv >nul 2>nul
    if not errorlevel 1 set UV_EXE=uv
)

if not defined UV_EXE (
    if exist "%PROJECT_DIR%.venv\Scripts\uv.exe" (
        set UV_EXE=%PROJECT_DIR%.venv\Scripts\uv.exe
    ) else (
        echo uv was installed, but it is not available in this command window yet.
        echo Open a new command window and run install.bat again.
        popd
        endlocal & exit /b 1
    )
)

set UV_CACHE_DIR=%PROJECT_DIR%.uv-cache
echo Installing OSRGen runtime dependencies...
"%UV_EXE%" sync --extra model
if errorlevel 1 (
    echo Dependency installation failed.
    popd
    endlocal & exit /b 1
)

echo.
echo Installation completed.
echo Start the GUI with launch_osrgen_gui.bat
popd
endlocal & exit /b 0
