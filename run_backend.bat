@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: Backend startup — Medical AI Cancer Detection
::
:: CUDA_VISIBLE_DEVICES=-1  forces CPU-only mode.  TF's CUDA scanner hangs
:: indefinitely on Windows when CUDA drivers are absent or mismatched.
:: This must be set at the OS level BEFORE Python starts — os.environ in
:: Python code runs too late because TF's C++ DLL scans for CUDA during load.
::
:: TF_ENABLE_ONEDNN_OPTS=0  disables the oneDNN floating-point warning message.
:: TF_CPP_MIN_LOG_LEVEL=3   silences TF's noisy C++ INFO/WARNING stderr lines.
:: ─────────────────────────────────────────────────────────────────────────────

set CUDA_VISIBLE_DEVICES=-1
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3

echo [backend] Starting uvicorn (CPU-only mode, no --reload) ...
echo [backend] API will be available at http://127.0.0.1:8000
echo [backend] Press Ctrl+C to stop.
echo.

call myvenv\Scripts\activate.bat

:: No --reload: avoids Windows double-process Ctrl+C trap.
:: Use --reload only during active development, never in production.
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
