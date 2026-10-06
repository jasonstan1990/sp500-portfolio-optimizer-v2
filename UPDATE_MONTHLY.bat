@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
 echo Install Python 3.11 or newer from python.org, then try again.
 pause
 exit /b 1
)
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r update_requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe update_data.py
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Update failed. Do not upload anything. See the error above.
pause
exit /b 1
