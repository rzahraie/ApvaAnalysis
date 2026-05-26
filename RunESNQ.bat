@echo off
setlocal

set REGIME=122022
set ESRAW=data\validation\ES\xApvaV01StateLog_ES%REGIME%.csv
set NQRAW=data\validation\NQ\xApvaV01StateLog_NQ%REGIME%.csv
set OUTCSV=tables\apva_forward_signed_return_dataset_es_nq_%REGIME%_generated.csv
set READINESS=outputs\dataset_readiness_es_nq_%REGIME%

echo Checking input files...
if not exist "%ESRAW%" (
    echo ERROR: Missing ES file: %ESRAW%
    pause
    exit /b 1
)

if not exist "%NQRAW%" (
    echo ERROR: Missing NQ file: %NQRAW%
    pause
    exit /b 1
)

echo.
echo Building canonical dataset...
python scripts\apva_build_forward_return_dataset.py ^
  --inputs "%ESRAW%" "%NQRAW%" ^
  --out "%OUTCSV%" ^
  --horizons 5 ^
  --compare-to "tables\apva_forward_signed_return_dataset_v1.csv"

if errorlevel 1 (
    echo ERROR: Dataset build failed.
    pause
    exit /b 1
)

echo.
echo Running readiness check...
python scripts\apva_dataset_readiness_check.py ^
  --inputs "%OUTCSV%" ^
  --outdir "%READINESS%"

if errorlevel 1 (
    echo ERROR: Readiness check failed.
    pause
    exit /b 1
)

echo.
echo Running fixed candidate extended validation...
python scripts\apva_fixed_candidate_extended_validation.py

if errorlevel 1 (
    echo ERROR: Extended validation failed.
    pause
    exit /b 1
)

echo.
echo DONE.
echo.
echo Generated table:
echo %OUTCSV%
echo.
echo Readiness outputs:
echo %READINESS%
echo.
echo Extended validation outputs:
echo outputs\fixed_candidate_extended_validation
echo.

pause
endlocal