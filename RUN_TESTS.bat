@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
 echo Install Python 3.11 or newer from python.org, then try again.
 pause
 exit /b 1
)
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r update_requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -v
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Tests failed. See the error above.
pause
exit /b 1
