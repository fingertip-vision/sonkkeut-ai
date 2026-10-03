$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
  if (Test-Path 'CTranslate2') { throw 'CTranslate2 already exists; use the prepared source bundle or a fresh directory.' }
  git clone --depth 1 --branch v4.8.2 https://github.com/OpenNMT/CTranslate2.git CTranslate2
  if ($LASTEXITCODE -ne 0) { throw 'Source download failed.' }
  $taskCommit = git -C CTranslate2 rev-parse HEAD
  if ($taskCommit -ne 'd44d2d069eb88c7b7804da864c10c201501cb4a9') { throw 'Unexpected CTranslate2 commit.' }
  git -C CTranslate2 -c core.longpaths=true submodule update --init --recursive --depth 1 third_party/cpu_features third_party/ruy third_party/spdlog
  if ($LASTEXITCODE -ne 0) { throw 'Pinned CPU dependencies download failed.' }
  git -C CTranslate2 apply ../patches/android-pthread-affinity.patch
  if ($LASTEXITCODE -ne 0) { throw 'Android affinity patch failed.' }
} finally { Pop-Location }
