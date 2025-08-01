@echo off
REM Docker helper script for LCA GNNs project (Windows)
REM Run this script from the project root directory

setlocal enabledelayedexpansion

REM Get the project root directory (parent of docker folder)
for %%i in ("%~dp0..") do set "PROJECT_ROOT=%%~fi"
cd /d "%PROJECT_ROOT%"

REM Check if Docker is installed and running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not running. Please install and start Docker.
    exit /b 1
)

REM Parse command
set COMMAND=%1

if "%COMMAND%"=="build" goto build
if "%COMMAND%"=="build-dev" goto build_dev
if "%COMMAND%"=="inference" goto inference
if "%COMMAND%"=="batch" goto batch
if "%COMMAND%"=="dev" goto dev
if "%COMMAND%"=="test" goto test
if "%COMMAND%"=="quality" goto quality
if "%COMMAND%"=="clean" goto clean
if "%COMMAND%"=="help" goto usage
if "%COMMAND%"=="-h" goto usage
if "%COMMAND%"=="--help" goto usage

echo [ERROR] Unknown command: %COMMAND%
goto usage

:build
echo [INFO] Building LCA GNNs Docker image...
docker build -f docker/Dockerfile -t lca-gnn:latest .
if errorlevel 1 exit /b 1
echo [INFO] Build completed successfully!
goto end

:build_dev
echo [INFO] Building LCA GNNs development Docker image...
docker build -f docker/Dockerfile.dev -t lca-gnn:dev .
if errorlevel 1 exit /b 1
echo [INFO] Development build completed successfully!
goto end

:inference
if "%2"=="" (
    echo [ERROR] Usage: %0 inference ^<model_path^> ^<smiles^> [additional_args...]
    echo Example: %0 inference trained_models/GNN_C_Gwi.pth "CCO" --target_task Gwi --country_name Germany
    exit /b 1
)

set MODEL_PATH=%2
set SMILES=%3
shift /2
shift /2
shift /2

echo [INFO] Running inference for SMILES: %SMILES%
docker run --rm ^
    -v "%cd%/trained_models:/app/trained_models:ro" ^
    -v "%cd%/data:/app/data:ro" ^
    lca-gnn:latest ^
    python main.py --workflow inference ^
    --model_path %MODEL_PATH% ^
    --smiles %SMILES% ^
    %4 %5 %6 %7 %8 %9
goto end

:batch
if "%3"=="" (
    echo [ERROR] Usage: %0 batch ^<model_path^> ^<data_path^> [additional_args...]
    echo Example: %0 batch trained_models/GNN_C_multi_best.pth test_molecules.xlsx
    exit /b 1
)

set MODEL_PATH=%2
set DATA_PATH=%3
shift /3
shift /3
shift /3

echo [INFO] Running batch processing for: %DATA_PATH%
docker run --rm ^
    -v "%cd%/trained_models:/app/trained_models:ro" ^
    -v "%cd%/data:/app/data" ^
    -v "%cd%/%DATA_PATH%:/app/%DATA_PATH%:ro" ^
    lca-gnn:latest ^
    python main.py --workflow batch ^
    --model_path %MODEL_PATH% ^
    --data_path %DATA_PATH% ^
    %4 %5 %6 %7 %8 %9
goto end

:dev
echo [INFO] Starting development environment...
docker run -it --rm ^
    -v "%cd%:/app" ^
    -p 8888:8888 ^
    -p 8000:8000 ^
    lca-gnn:dev
goto end

:test
echo [INFO] Running tests in Docker container...
docker run --rm ^
    -v "%cd%:/app" ^
    lca-gnn:dev ^
    bash -c "cd /app && python -m pytest tests/ -v"
goto end

:quality
echo [INFO] Running code quality checks in Docker container...
docker run --rm ^
    -v "%cd%:/app" ^
    lca-gnn:dev ^
    bash -c "cd /app && python -m ruff check src/ tests/ && python -m ruff format --check src/ tests/ && python -m pyright src/"
goto end

:clean
echo [INFO] Cleaning up Docker resources...
docker system prune -f
echo [INFO] Cleanup completed!
goto end

:usage
echo LCA GNNs Docker Helper Script (Windows)
echo.
echo Usage: %0 ^<command^> [arguments...]
echo.
echo Commands:
echo   build                    Build the production Docker image
echo   build-dev                Build the development Docker image
echo   inference ^<model^> ^<smiles^> [args...]  Run single molecule inference
echo   batch ^<model^> ^<data^> [args...]        Run batch processing
echo   dev                      Start development environment
echo   test                     Run tests in container
echo   quality                  Run code quality checks
echo   clean                    Clean up Docker resources
echo.
echo Examples:
echo   %0 build
echo   %0 inference trained_models/GNN_C_Gwi.pth "CCO" --target_task Gwi
echo   %0 batch trained_models/GNN_C_multi_best.pth test_data.xlsx
echo   %0 dev
goto end

:end
endlocal
