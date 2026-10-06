@echo off
chcp 65001 >nul
cd /d "%~dp0"
call _env.bat
if not defined PY (
  echo [1/4] 가상환경 만드는 중 .venv
  py -3.11 -m venv .venv || python -m venv .venv
  set "PY=%~dp0.venv\Scripts\python.exe"
  "%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
  echo [2/4] GPU용 PyTorch 설치 중 - RTX 50 시리즈는 cu128
  "%~dp0.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  call _env.bat
) else (
  echo [1/4] 기존 가상환경 사용: %PY%
  echo [2/4] PyTorch는 이미 설치되어 있음
)
echo [3/4] 패키지 설치 중
"%PY%" -m pip install -r requirements.txt
echo [4/4] 손 관절 모델 내려받기 + 테스트
"%PY%" -c "from sonkkeut_vision.fingertip import ensure_hand_model; print(ensure_hand_model())"
"%PY%" -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available())"
"%PY%" -m pytest -q
echo.
echo 끝났습니다. simulate.bat 으로 시뮬레이션, run_camera.bat 으로 웹캠 데모를 실행하세요.
pause
