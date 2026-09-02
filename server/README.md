# 인식 성공/실패 자료 수신 서버

클라이언트가 사용자에게 정답을 제공한 세션과 제공하지 못한 세션을 HTTPS로 받아 서로 다른 경로에 보관합니다. 수신기는 Python 표준 라이브러리만 사용하며 Nginx 뒤의 `127.0.0.1:8765`에서 실행합니다.

## 운영 구성

- Oracle 인스턴스: `gta-report-receiver` (`VM.Standard.E2.1.Micro`, 50GB 부트 볼륨)
- 네트워크: DC 매니저와 분리된 `10.0.1.0/24` 전용 서브넷과 전용 보안 목록
- 공개 주소: `https://gta-reports.64-110-118-28.sslip.io/v1/reports`
- 상태 확인: `https://gta-reports.64-110-118-28.sslip.io/health`
- 프로그램: `/opt/gta-report-receiver/receiver.py`
- 성공 저장 경로: `/var/lib/gta-report-receiver/success/YYYY-MM-DD/`
- 실패 저장 경로: `/var/lib/gta-report-receiver/failure/YYYY-MM-DD/`
- 서비스: `gta-report-receiver.service`
- 정리 타이머: `gta-report-cleanup.timer`
- 보존 기간: 30일
- 전체 저장 한도: 5GB

이 서버는 Oracle Always Free의 두 번째 E2.1.Micro와 전체 무료 200GB 중 50GB 부트 볼륨을 사용합니다. 유료 로드밸런서, Object Storage, 별도 블록 볼륨과 유료 백업을 사용하지 않습니다.

수신 ZIP은 `session.json`과 세션 JPEG만 허용합니다. `session.json`의 답 제공 결과를 확인하고, 이전 버전 자료는 정답 요약·신뢰도·기준값으로 호환 분류합니다. 경로가 포함된 파일, 중복 이름, 비정상 JPEG, 50MB 초과 요청과 100MB 초과 압축 해제 크기는 거부합니다. Nginx는 요청 크기와 IP별 요청 속도를 제한하며 업로드 경로의 접근 로그는 남기지 않습니다.

## 확인 명령

```bash
systemctl is-active gta-report-receiver.service
systemctl is-active gta-report-cleanup.timer
systemctl is-active nginx
curl --fail https://gta-reports.64-110-118-28.sslip.io/health
journalctl -u gta-report-receiver.service -n 30 --no-pager
```

인증서는 Certbot 타이머가 자동 갱신합니다. 설정 변경 후에는 `nginx -t`가 성공한 경우에만 Nginx를 다시 불러옵니다.
