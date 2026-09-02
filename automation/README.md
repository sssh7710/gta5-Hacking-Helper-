# 예약 개선 작업 이력

두 컴퓨터가 서로 다른 시간에 진단 수집 서버의 새 자료를 순서대로 처리할 때 사용합니다. 실제 진단 이미지나 서버 접속 정보는 이 폴더에 저장하지 않습니다.

`improvement-history.jsonl`은 첫 성공 기록부터 자동 생성됩니다. OneDrive로 두 컴퓨터의 작업 폴더가 동기화되어 있어야 하며, 다음 예약 작업 전 OneDrive 동기화가 끝났는지 확인합니다.

각 작업은 서버 자료를 반열린 구간 `[server_log_from_utc, server_log_to_utc)`으로 조회합니다. 즉 시작 시각은 포함하고 종료 시각은 제외합니다. 다음 성공 작업의 시작 시각은 직전 성공 작업의 종료 시각과 같으므로, 구간이 겹치거나 비지 않습니다. 서버 파일의 수신 시각은 UTC로 비교합니다.

## 주간 작업 순서

1. 다음 조회 구간을 확인합니다. 첫 작업은 진단 자료를 처리하기 시작할 기준 시각을 직접 지정합니다.

```powershell
.\.venv\Scripts\python.exe -B tools\improvement_history.py next `
  --initial-from-utc 2026-09-01T00:00:00Z
```

2. 출력된 시작·종료 시각 범위의 서버 진단 자료만 가져와 분석·개선합니다.
3. 전체 테스트를 실행하고 개선 커밋을 만듭니다.
4. 성공한 경우에만 같은 구간, 가져온 자료 수, PC 이름, 커밋 SHA를 기록합니다.

```powershell
.\.venv\Scripts\python.exe -B tools\improvement_history.py record `
  --computer PC-A `
  --from-utc 2026-09-01T00:00:00Z `
  --to-utc 2026-09-08T01:00:00Z `
  --report-count 24 `
  --commit abc1234 `
  --summary "점멸 인식 회귀 보정"
```

실패한 작업도 `--result failure`로 남길 수 있지만, 실패 기록은 다음 작업의 조회 시작 시각을 앞으로 옮기지 않습니다. 실패한 구간은 다음 예약 작업에서 다시 처리합니다.

## 월간 릴리스

매월 `improvement-history.jsonl`의 성공 기록과 해당 커밋을 확인한 뒤, 누적 커밋을 `dev` 브랜치에 푸시하고 새 베타 태그·전체 파일 ZIP·SHA-256 체크섬으로 GitHub Release를 만듭니다.
