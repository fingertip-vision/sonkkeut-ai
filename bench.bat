@echo off
chcp 65001 >nul
cd /d "%~dp0"
call _env.bat
if not defined PY (echo setup.bat 을 먼저 실행하세요 & pause & exit /b 1)
"%PY%" -m pytest -q
set "M1DATA=%USERPROFILE%\Documents\sonkkeut\data\m1"
if exist "%M1DATA%\images\val" ("%PY%" scripts\bench.py --m1-data "%M1DATA%") else ("%PY%" scripts\bench.py)
"%PY%" scripts\eval_scenarios.py --n 40
pause
