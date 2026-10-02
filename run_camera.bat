@echo off
chcp 65001 >nul
cd /d "%~dp0"
call _env.bat
if not defined PY (echo setup.bat 을 먼저 실행하세요 & pause & exit /b 1)
rem 카메라 번호를 바꾸려면: run_camera.bat --source 1
"%PY%" scripts\run_camera.py --tts %*
pause
