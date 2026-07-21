# GTA V Enhanced 해킹 안내 도우미

GTA Online 해킹 화면을 **외부 화면 캡처만으로 분석**해 정답 위치를 알려주는 Windows용 도구입니다. 게임에 키·마우스 입력을 보내지 않고, 게임 메모리·프로세스·DLL·DirectX를 건드리지 않습니다.

## 현재 지원 대상

- 코르츠 센터/카지노 계열 점멸 원 기억 퍼즐: 같은 줄을 묶어 `2번째 줄: 3번, 5번` 형태로 정답 칸 표시
- 카지노·코르츠 지문 조각: 정답 후보 4개의 번호 표시
- 카요 페리코 지문 조립: 줄별 정답 조각과 최소 이동 방향 표시

카지노 지문 UI는 [사용자 제공 연습 영상](https://youtu.be/VDEClygt3hc)의 00:07~00:34 구간으로 레이아웃을 보정했습니다. 실제 GTA UI 캡처 자료는 포함하지 않으므로, 첫 실행 전후에 `진단 저장`으로 본인 화면을 수집해 인식 결과를 확인해야 합니다. 지문 정답표는 `assets/reference/`에 넣어 회귀 테스트 자료로 관리할 수 있습니다. 실행 중에는 정답표 파일을 사용하지 않습니다.

## 설치와 실행

1. Windows에서 [Python 3.13](https://www.python.org/downloads/)을 설치합니다.
2. [setup.bat](setup.bat)을 한 번 실행합니다.
3. [run.bat](run.bat)을 실행합니다.

`setup.bat`은 `.venv`에 DXcam, OpenCV, NumPy, 음성 라이브러리를 설치합니다.

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

테스트는 점멸 패턴 반복 확정, 지문 조각 선택, 카요 줄별 이동 계산, 설정 저장을 합성 이미지로 검증합니다.

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
