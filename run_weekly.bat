@echo off
REM Weekly digest, for Windows Task Scheduler.
REM A .bat wrapper rather than a bare schtasks /TR command: quoting a path with
REM spaces through schtasks is fragile, and this also pins the working directory
REM so relative paths in config.py resolve.

cd /d "%~dp0"
".venv\Scripts\python.exe" run_digest.py --log
exit /b %ERRORLEVEL%
