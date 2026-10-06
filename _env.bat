@echo off
rem 사용할 파이썬을 찾는다: (1) 이 폴더의 .venv (2) 예전에 학습에 쓴 Documents\sonkkeut\.venv
set "HERE=%~dp0"
set "PY="
if exist "%HERE%.venv\Scripts\python.exe" set "PY=%HERE%.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\Documents\sonkkeut\.venv\Scripts\python.exe" set "PY=%USERPROFILE%\Documents\sonkkeut\.venv\Scripts\python.exe"
