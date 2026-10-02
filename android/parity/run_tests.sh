#!/bin/bash
# 파이썬(기준) ↔ Kotlin(앱) 동등성 테스트 + Kotlin 파이프라인 시뮬레이션
# 필요: Python 패키지(requirements.txt), JDK 17+, Kotlin 컴파일러(kotlinc) 또는 Gradle 8 설치본
#   bash android/parity/run_tests.sh
set -e
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
SRC=android/react-native-sonkkeut/android/src/main/java/kr/sonkkeut/core
OUT=$(mktemp -d)

if [ ! -f android/parity/golden/guide.json ]; then
  echo "[1/3] 파이썬 정답 데이터 만드는 중"
  python android/parity/make_golden.py
fi

# Kotlin 컴파일러: kotlinc가 있으면 쓰고, 없으면 Gradle 설치본에 들어 있는 컴파일러를 쓴다
GL=$(dirname "$(readlink -f "$(command -v gradle)")")/../lib
STD=$(ls $GL/kotlin-stdlib-2*.jar 2>/dev/null | head -1)
SER=$(ls $GL/kotlinx-serialization-core-jvm-*.jar | head -1):$(ls $GL/kotlinx-serialization-json-jvm-*.jar | head -1)
kc() {
  if command -v kotlinc >/dev/null; then kotlinc -cp "$SER" -d "$OUT" "$@" 2>&1 | grep -v warning || true
  else
    CP=$(ls $GL/kotlin-compiler-embeddable-*.jar $GL/kotlin-stdlib-2*.jar $GL/kotlin-reflect-*.jar $GL/kotlin-script-runtime-*.jar $GL/trove4j-*.jar $GL/annotations-*.jar $GL/kotlin-daemon-embeddable-*.jar $GL/kotlinx-coroutines-core-jvm-*.jar | tr '\n' ':')
    java -cp "$CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -cp "$STD:$SER" -d "$OUT" -jvm-target 17 "$@"
  fi
}
echo "[2/3] Kotlin 컴파일"
kc $SRC/*.kt android/parity/ParityTest.kt android/parity/PipelineSimTest.kt
echo "[3/3] 테스트"
java -Dstdout.encoding=UTF-8 -cp "$OUT:$STD:$SER" ParityTestKt android/parity/golden
java -Dstdout.encoding=UTF-8 -cp "$OUT:$STD" PipelineSimTestKt
