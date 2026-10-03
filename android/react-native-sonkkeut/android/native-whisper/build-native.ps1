param([string]$AndroidSdk = (Join-Path $PSScriptRoot '../android-sdk'))
$ErrorActionPreference = 'Stop'
$taskSdk = (Resolve-Path $AndroidSdk).Path.Replace('\', '/')
$taskCmake = "$taskSdk/cmake/3.22.1/bin/cmake.exe"
$taskNinja = "$taskSdk/cmake/3.22.1/bin/ninja.exe"
$taskNdk = "$taskSdk/ndk/26.1.10909125"
$common = @('-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$taskNinja",
  "-DCMAKE_TOOLCHAIN_FILE=$taskNdk/build/cmake/android.toolchain.cmake",
  '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-24', '-DANDROID_STL=c++_shared', '-DCMAKE_BUILD_TYPE=Release')
Push-Location $PSScriptRoot
try {
  if (!(Test-Path 'CTranslate2/CMakeLists.txt')) { throw 'Unpack the pinned source bundle or run fetch-source.ps1 first.' }
  & $taskCmake -S CTranslate2 -B build-arm64 @common '-DCMAKE_POSITION_INDEPENDENT_CODE=ON' '-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384' '-DWITH_RUY=ON' '-DWITH_MKL=OFF' '-DWITH_DNNL=OFF' '-DWITH_OPENBLAS=OFF' '-DWITH_ACCELERATE=OFF' '-DWITH_CUDA=OFF' '-DWITH_CUDNN=OFF' '-DWITH_HIP=OFF' '-DOPENMP_RUNTIME=NONE' '-DENABLE_CPU_DISPATCH=OFF' '-DBUILD_CLI=OFF' '-DBUILD_TESTS=OFF' '-DBUILD_SHARED_LIBS=ON'
  if ($LASTEXITCODE -ne 0) { throw 'CTranslate2 configuration failed.' }
  & $taskCmake --build build-arm64 --parallel 4
  if ($LASTEXITCODE -ne 0) { throw 'CTranslate2 compilation failed.' }
  & $taskCmake -S . -B build-jni-arm64 @common
  if ($LASTEXITCODE -ne 0) { throw 'JNI configuration failed.' }
  & $taskCmake --build build-jni-arm64 --parallel 4
  if ($LASTEXITCODE -ne 0) { throw 'JNI compilation failed.' }
  New-Item -ItemType Directory -Path 'jniLibs/arm64-v8a' -Force | Out-Null
  Copy-Item -LiteralPath 'build-arm64/libctranslate2.so','build-jni-arm64/libsonkkeut_whisper.so',"$taskNdk/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" -Destination 'jniLibs/arm64-v8a'
  Get-ChildItem -LiteralPath 'jniLibs/arm64-v8a' -Filter '*.so' | ForEach-Object { & "$taskNdk/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe" --strip-unneeded $_.FullName }
} finally { Pop-Location }
