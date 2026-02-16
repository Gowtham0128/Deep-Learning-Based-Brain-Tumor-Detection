@echo off
REM Activate the venv if present and run the default processing script
SET ROOT_DIR=%~dp0
IF EXIST "%ROOT_DIR%\.venv\Scripts\activate.bat" (
  CALL "%ROOT_DIR%\.venv\Scripts\activate.bat"
)
python "%ROOT_DIR%process_all_images.py"
EXIT /B %ERRORLEVEL%
