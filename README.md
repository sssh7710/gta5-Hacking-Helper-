# GTA V Enhanced 해킹 안내 도우미 (개발중)

GTA Online 해킹 화면을 **외부 화면 캡처만으로 분석**해 정답 위치를 알려주는 Windows용 도구입니다. 게임에 키·마우스 입력을 보내지 않고, 게임 메모리·프로세스·DLL·DirectX를 건드리지 않습니다.

## 릴리스 상태

- 코르츠 센터 습격 점멸 키패드 해킹: 멘션 실게임에서 4단계 연속 인식 확인
- 아케이드 연습 장비의 점멸 키패드: 현재 화면 레이아웃 차이로 인식 불가
- 카지노·코르츠·카요 지문 인식: 현재 개발·보정 중이며 정답 정확도를 보장하지 않음

## 현재 지원 대상

- 코르츠 센터 습격 점멸 원 기억 퍼즐: 현재 멘션 화면만 지원하며, 보통은 01~05(5열×4행), 어려움은 01~06(6열×5행) 순서와 세로 위치를 표시(아케이드 미지원)
- 카지노·코르츠 지문 조각(개발 중): 정답 후보 4개의 번호 표시
- 카요 페리코 지문 조립(개발 중): 줄별 정답 조각과 최소 이동 방향 표시

카지노 지문 UI는 [사용자 제공 연습 영상](https://youtu.be/VDEClygt3hc)의 00:07~00:34 구간과 1920×1080 실캡처로 레이아웃을 보정했습니다. 실행 중에는 전체 대상 지문 안에서 후보 타일의 테두리를 제외한 무늬를 다중 크기로 직접 비교하며, 정답표 이미지를 읽지 않습니다. 첫 실행 전후에는 `진단 저장`으로 본인 화면을 수집해 인식 결과를 확인하세요. 지문 정답표는 `assets/reference/`에 넣어 회귀 테스트 자료로 관리할 수 있습니다.

## 설치와 실행

1. Windows에서 [Python 3.13](https://www.python.org/downloads/)을 설치합니다. 설치 화면에서 `Add python.exe to PATH`를 켜는 것을 권장합니다.
2. 압축을 완전히 푼 폴더에서 [run.bat](run.bat)을 실행합니다.
3. 첫 실행이면 `run.bat`이 [setup.bat](setup.bat)을 호출해 `.venv`와 필수 패키지를 자동으로 설치합니다.

`setup.bat`은 `.venv`에 DXcam, OpenCV, NumPy, 음성 라이브러리를 설치합니다.

### 다른 PC에서 창이 바로 닫힐 때

- `app.py`를 직접 열지 말고 반드시 `run.bat`을 실행하세요.
- 수정된 `run.bat`은 앱이 오류로 종료되면 콘솔을 닫지 않고 오류 메시지를 표시합니다.
- 다른 PC에서 복사해 온 `.venv`는 사용할 수 없습니다. 프로젝트 폴더의 `.venv`만 삭제한 뒤 `run.bat`을 다시 실행하세요.
- 압축 파일 내부에서 직접 실행하지 말고 쓰기 가능한 일반 폴더에 모두 압축 해제하세요.

## 화면 모드와 안내 방식

- `클릭 통과 오버레이`: 기본 모드입니다. 위치를 먼저 조정한 뒤 `오버레이 잠금`을 누르면 마우스 클릭이 게임에 통과합니다.
- `일반 작은 창`: 창 위치와 크기를 자유롭게 조정할 수 있습니다.
- `음성 전용`: 창을 작업 표시줄로 최소화하고 음성으로만 답을 안내합니다. 음성은 설정에서 켜야 합니다.

독점 전체 화면에서는 Windows·GPU 환경에 따라 외부 오버레이가 보이지 않을 수 있습니다. 단일 모니터에서 시각 안내가 필요하면 GTA를 **테두리 없는 창**으로 실행하세요. 인식 자체는 DXcam의 DXGI/WinRT 백엔드를 순서대로 시도합니다.

## 설정

첫 실행 시 생성되는 `config.json`에서 캡처 백엔드, 출력 모니터, 안내 위치·투명도, 음성, 키보드 범례를 바꿀 수 있습니다. 설정 창에서 바꾼 캡처·음성 설정은 다음 실행부터 적용됩니다.

기본 키보드 범례는 방향키/WASD, Enter/마우스 1, Backspace/Esc입니다. Xbox·PlayStation 범례와 개인 키 매핑은 다음 UI 보정 단계에서 확장할 수 있도록 `custom_keys`에 저장됩니다.

## 검증

```powershell
python -B -m unittest discover -s tests -v
```

테스트는 점멸 패턴 반복 확정, 고해상도 지문 영역의 상대 좌표 절단, 지문 조각 선택, 카요 줄별 이동 계산, 설정 저장을 합성 이미지로 검증합니다.

## 참조 영상 프레임 추출

실전 영상의 UI 변화를 확인할 때는 1초 간격 PNG를 만듭니다.

```powershell
.venv\Scripts\python.exe -B tools\extract_video_frames.py `
  diagnostics\fingerprint_reference.mp4 diagnostics\fingerprint_frames `
  --start 7 --end 34 --interval 1
```

완성된 4개 조각 선택의 시간·번호를 회귀 자료로 기록하려면 다음 명령을 사용합니다. 이 도구는 영상의 밝은 선택 표시만 읽으며 게임을 조작하지 않습니다.

```powershell
.venv\Scripts\python.exe -B tools\analyze_casino_video.py `
  diagnostics\fingerprint_reference.mp4 assets\reference\casino_video_labels.json `
  --start 7 --end 34
```

수집한 완료 조합으로 실행용 템플릿을 만들 수 있습니다. 이 템플릿은 정답표 사진을 실행 중에 읽지 않으며, 신뢰도가 낮은 다른 지문 유형에는 답을 표시하지 않습니다.

```powershell
.venv\Scripts\python.exe -B tools\build_casino_templates.py `
  diagnostics\fingerprint_reference.mp4 assets\reference\casino_video_labels.json `
  assets\reference\casino_templates.json
```

## 주의

이 프로그램은 자동 조작을 하지 않지만, 온라인 게임의 제3자 도구 관련 정책·제재 여부를 보장하지 않습니다. 사용 전 Rockstar 정책을 직접 확인하세요.
