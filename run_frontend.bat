@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: Frontend startup — Medical AI Cancer Detection
:: Run this in a SEPARATE terminal after run_backend.bat is ready.
:: ─────────────────────────────────────────────────────────────────────────────

echo [frontend] Starting Streamlit ...
echo [frontend] UI will be available at http://localhost:8501
echo [frontend] Make sure the backend is already running on port 8000.
echo [frontend] Press Ctrl+C to stop.
echo.

call myvenv\Scripts\activate.bat

streamlit run frontend/streamlit_app.py
