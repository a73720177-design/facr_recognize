# 얼굴 식별·동작 패턴 2단계 인증 금고

등록된 얼굴을 확인한 다음 그 사용자에게 저장된 손·팔 자세를 순서대로 비교하는 Python 프로그램입니다. 실제 잠금장치는 연결하지 않았습니다.

**1~3번 금고를 제공하며, 금고 하나당 소유자 한 명·사용자 한 명당 금고 하나만 허용합니다.** 터미널에서 번호를 선택하거나 `--safe-number 2`로 지정하세요. 창 제목과 화면에 선택 번호, 전체 사용 상태, 얼굴 확인된 사용자의 금고 번호를 표시합니다. 다른 사람의 금고는 사용할 수 없으며, 소유자가 얼굴·동작 인증 후 R을 5초 안에 두 번 눌러 해당 번호를 초기화해야 새 사용자가 등록할 수 있습니다.

## 실행

```powershell
cd C:\angchicken
.\.venv\Scripts\python.exe recognition.py
```

하나의 메뉴에서 **1: 금고 등록 / 2: 잠금 해제 / 0: 종료**를 선택합니다. 두 작업 모두 다음에 **금고 번호(1~3)**를 입력합니다. 등록은 이름·단계 수를 입력하고, 새 사용자라면 얼굴 저장부터 동작 촬영까지 이어집니다. 각 동작에서 P를 누르면 5초 뒤 촬영합니다. 잠금 해제는 얼굴 확인 후 U로 동작 비교를 시작합니다. 카메라에서 M 또는 Q/Esc를 누르면 같은 프로그램의 메뉴로 돌아와 다른 번호나 작업을 고를 수 있습니다. 등록 완료·초기화 후에도 메뉴로 돌아갑니다.

```powershell
# 사용자 얼굴 등록: 카메라 창에서 S로 저장
.\.venv\Scripts\python.exe recognition.py --register "홍길동"
# 동작 패턴 등록: 터미널에서 단계 수 입력
.\.venv\Scripts\python.exe recognition.py --safe-mode enroll
# 잠금 해제 시도
.\.venv\Scripts\python.exe recognition.py --safe-mode verify
# 2번 금고 선택
.\.venv\Scripts\python.exe recognition.py --safe-mode verify --safe-number 2
# 모델 점검과 테스트
.\.venv\Scripts\python.exe recognition.py --self-test
.\.venv\Scripts\python.exe -m unittest test_safe_menu test_safe_pattern test_safe_slot -v
```

상세 안내는 [금고 사용 설명](SAFE_PATTERN_GUIDE.md)을 참고하세요. 이전 [입문자 문서](BEGINNER_GUIDE.md), [프로젝트 작업 이력](PROJECT_SUMMARY.md), [금고 적용 학습 기록](SAFE_PROJECT_LEARNINGS.md)은 과거 실습 내용도 포함합니다.

## 설치

Windows / Python 3.14.7 환경에서 검증했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe recognition.py --self-test
```

모델은 최초 실행 시 다운로드하며 이후 영상 처리는 로컬에서 수행합니다. 한글 표시는 Windows 맑은 고딕을 사용합니다.

## 현재 파일 구성

| 파일 | 역할 |
| --- | --- |
| `recognition.py` | 실행 옵션, 얼굴 등록, 모델 실행 점검 |
| `safe_app.py` | 터미널 입력·카메라·2단계 인증 화면 |
| `face_identity.py` | YuNet/SFace 얼굴 식별과 등록 자료 관리 |
| `face_gate.py` | 얼굴 식별 안정화·사용자 변경·소실 처리 |
| `safe_pattern.py` | 손·팔 특징 추출, 단계별 촬영·저장·비교 |
| `safe_slot.py` | 금고 번호·1인 1금고 소유권·초기화·동시 변경 보호 |
| `vision_utils.py` | 손·신체 모델, 관절 표시, 한글 렌더링 |
| `test_safe_pattern.py` | 패턴과 인증 흐름 테스트 |
| `test_safe_slot.py` | 소유권·등록 제한·초기화·기존 패턴 이전 테스트 |
| `test_safe_menu.py` | 통합 메뉴의 등록→해제 전환·번호 선택·오류 복귀 테스트 |
| `run.ps1` | 가상환경 Python 실행 |

현재 사용하지 않는 표정·얼굴 비율·가위바위보 이름 판별·손 흔들기·블러 기능과 `--demo`, `--hide-hand-ids`, `--pattern-steps` 옵션을 삭제했습니다. 손·팔의 자세 수치 자체를 비교하며 표정은 사용하지 않습니다.

## 데이터와 제한

얼굴 자료는 `data/faces.json`, 1~3번의 소유자·패턴은 `data/safe_slot.json`에 함께 저장합니다. 기존 단일 공간의 소유자와 패턴은 1번으로 유지합니다. 초기화는 선택 번호에만 적용하며 다른 금고는 유지합니다. 가상환경·모델·등록 데이터는 Git에서 제외합니다.

현재는 카메라에 보이는 정지 자세의 순서를 비교합니다. 얼굴의 생체 확인, 얼굴과 손의 동일 인물 연결, 관리자 등록 보호와 실제 금고 제어는 별도 구현이 필요합니다. 카메라를 고정하고 비슷한 거리와 조명에서 사용하세요.
