#!/bin/bash
# 손끝길용 AIHub 데이터 받기 (이어받기·재시도·분할파일 병합)
# 사용: AIHUB_APIKEY=<본인 키> bash /d/sonkkeutgil/aihub/download.sh [ds:filekey ...]   (인자 없으면 기본 6개)
#   71553 관광 음식메뉴판   Validation 광주 원천(899MB) + 라벨  -> M3 실사진 평가/학습
#   102   소상공인 고객 주문 질의-응답  train+val (188MB)        -> F-06 주문 이해
#   105   야외 실제 촬영 한글 이미지  추가분 Validation VS1(3GB)+VL1 -> M3 실촬영 열화
cd "$(dirname "$0")"
[ -n "$AIHUB_APIKEY" ] || { echo "AIHUB_APIKEY 를 설정하세요"; exit 1; }
JOBS=${*:-"71553:485598 71553:485611 102:44059 102:44060 105:68072 105:68073"}
touch done.txt
for job in $JOBS; do
  ds=${job%%:*}; k=${job##*:}
  grep -qx "$job" done.txt && continue
  mkdir -p "$ds" && pushd "$ds" >/dev/null
  ok=0
  for try in 1 2 3 4 5; do
    echo "[$(date +%T)] $ds/$k try $try"
    code=$(curl -sS -L -C - -o "dl_$k.tar" -H "apikey:$AIHUB_APIKEY" -w "%{http_code}" \
           "https://api.aihub.or.kr/down/0.6/$ds.do?fileSn=$k")
    if [ "$code" = "200" ] || [ "$code" = "206" ]; then
      if tar -xf "dl_$k.tar"; then ok=1; break; fi
      echo "tar 실패, 재다운로드"; rm -f "dl_$k.tar"
    else
      echo "HTTP $code: $(head -c 300 "dl_$k.tar" 2>/dev/null)"; rm -f "dl_$k.tar"
      [ "$code" = "401" ] || [ "$code" = "403" ] && { echo "권한 없음(이용 신청 필요?) -> 건너뜀"; break; }
      sleep 30
    fi
  done
  if [ $ok = 1 ]; then
    # 큰 파일은 xxx.zip.part0, part1... 로 쪼개져 온다 -> 합치기
    find . -name "*.part0" | while read -r p0; do
      base=${p0%.part0}
      cat $(ls -v "$base".part*) > "$base" && rm -f "$base".part*
      echo "병합: $base"
    done
    rm -f "dl_$k.tar"
    echo "$job" >> ../done.txt
  else
    echo "FAIL $job" | tee -a ../fail.txt
  fi
  popd >/dev/null
done
echo "끝. 받은 것:"; cat done.txt; [ -f fail.txt ] && { echo "실패:"; cat fail.txt; } || true
