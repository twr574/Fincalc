@echo off
for /f "usebackq delims=" %%a in ("%~dp0env_path.cfg") do set "ENV_PATH=%%a"
set "SCRIPT_PATH=%~dp0demo.py"
set "PATH=%PATH%;%ENV_PATH%"
python "%SCRIPT_PATH%"