@echo off
chcp 65001 >nul
cd /d "%~dp0"
call _env.bat
if not defined PY (echo setup.bat 을 먼저 실행하세요 & pause & exit /b 1)
"%PY%" scripts\simulate.py --out sim_demo.mp4 %*
start "" sim_demo.mp4
pause
