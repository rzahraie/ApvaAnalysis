@echo off
setlocal

REM ============================================================
REM APVA Phase 2A - 2020 Regime Validation Pipeline
REM ============================================================

REM ------------------------------------------------------------
REM STEP 1 - Build canonical forward-return dataset
REM ------------------------------------------------------------

python scripts\apva_build_forward_return_dataset.py ^
  --inputs ^
  "data\validation\ES\xApvaV01StateLog_ES092020.csv" ^
  "data\validation\NQ\xApvaV01StateLog_NQ092020.csv" ^
  --out ^
  "tables\apva_forward_signed_return_dataset_es_nq_092020_generated.csv" ^
  --horizons 5 ^
  --compare-to ^
  "tables\apva_forward_signed_return_dataset_v1.csv"

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Dataset build failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Dataset build complete.
echo ============================================================
echo.

REM ------------------------------------------------------------
REM STEP 2 - Readiness check
REM ------------------------------------------------------------

python scripts\apva_dataset_readiness_check.py ^
  --inputs ^
  "tables\apva_forward_signed_return_dataset_es_nq_092020_generated.csv" ^
  --outdir ^
  "outputs\dataset_readiness_es_nq_092020"

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Readiness check failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Readiness check complete.
echo ============================================================
echo.

REM ------------------------------------------------------------
REM STEP 3 - Fixed candidate extended validation
REM ------------------------------------------------------------

python scripts\apva_fixed_candidate_extended_validation.py

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Extended validation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Extended validation complete.
echo ============================================================
echo.

REM ------------------------------------------------------------
REM STEP 4 - Display important output locations
REM ------------------------------------------------------------

echo Readiness outputs:
echo outputs\dataset_readiness_es_nq_092020
echo.

echo Extended validation outputs:
echo outputs\fixed_candidate_extended_validation
echo.

echo Most important files:
echo outputs\fixed_candidate_extended_validation\extended_scorecard.csv
echo outputs\fixed_candidate_extended_validation\extended_candidate_summary.csv
echo.

pause
endlocal