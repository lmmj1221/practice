# 딥러닝 기반 정책 시계열 예측 시스템

## 프로젝트 구성

- `education.py`: 딥러닝 개념 교육용 시각화 프로그램
- `analysis.py`: 실제 데이터 분석 및 모델 학습 프로그램
- `auto_run.py`: 자동 실행 스크립트

---

## 🍎 macOS 사용자

### 환경 설정

#### 방법 1: 자동 설정 (권장) ✨
```bash
# 터미널에서 실행
source activate_venv.sh
```

#### 방법 2: 수동 설정
```bash
# 1. 가상환경 생성
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt
```

### 프로그램 실행
```bash
# macOS에서 직접 실행
python3 education.py   # 교육 프로그램
python3 analysis.py    # 분석 프로그램

# 가상환경 활성화 후
python education.py   # 교육 프로그램
python analysis.py    # 분석 프로그램
```

### 가상환경 비활성화
```bash
deactivate
```

---

## 🪟 Windows 사용자

### 환경 설정

#### 방법 1: PowerShell 사용
```powershell
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 3. 패키지 설치
pip install -r requirements.txt
```

#### 방법 2: Command Prompt 사용
```cmd
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
venv\Scripts\activate.bat

# 3. 패키지 설치
pip install -r requirements.txt
```

### 프로그램 실행
```cmd
# 가상환경 활성화 후
python education.py   # 교육 프로그램
python analysis.py    # 분석 프로그램
python auto_run.py    # 자동 실행
```

### 가상환경 비활성화
```cmd
deactivate
```

### Windows 추가 설정

#### PowerShell 실행 정책 오류 시
```powershell
# 관리자 권한으로 PowerShell 실행 후
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Python이 인식되지 않을 때
1. Python 공식 사이트에서 설치: https://www.python.org/
2. 설치 시 "Add Python to PATH" 체크박스 선택

---

## 📱 프로그램 사용법

### education.py - 교육용 시각화
실행 후 메뉴 선택:
1. 신경망 구조 시각화
2. 시계열 분석 개념
3. RNN 구조 설명
4. LSTM/GRU 비교
5. 전체 실행 (모든 시각화)

### analysis.py - 데이터 분석
실행 후 메뉴 선택:
1. 모델 학습 및 저장
2. 저장된 모델 로드 및 평가
3. 정책 영향 분석 (모델 학습 포함)
4. 정책 영향 분석 (저장된 모델 사용)
5. 통계적 분석
6. 전체 분석 실행 (새로 학습)

---

## 💻 시스템 요구사항

### 공통 요구사항
- Python 3.8 이상 (권장: 3.13.7)
- 메모리: 4GB 이상
- 저장공간: 2GB 이상

### OS별 지원
- **macOS**: 10.15 이상 (M1/M2 ARM64 지원)
- **Windows**: Windows 10/11 (64bit)
- **Linux**: Ubuntu 20.04 이상

## 📦 주요 패키지

- TensorFlow 2.20.0
- NumPy 2.3.3
- Pandas 2.3.2
- Matplotlib 3.10.6
- Scikit-learn 1.7.2
- Seaborn 0.13.2
- Statsmodels 0.14.5

---

## 🔧 문제 해결

### 공통 문제

#### Q: ModuleNotFoundError 오류
```bash
# 가상환경 활성화 확인 후 패키지 재설치
pip install --upgrade pip
pip install -r requirements.txt
```

#### Q: 메모리 부족 오류
- 모델 학습 시 batch_size를 16 또는 8로 줄이기
- analysis.py에서 옵션 1, 4만 실행 (모델 학습 제외)

### macOS 문제

#### Q: 한글 폰트 깨짐
- 자동으로 AppleGothic 폰트 적용됨
- 추가 설정 불필요

#### Q: permission denied
```bash
chmod +x activate_venv.sh
```

### Windows 문제

#### Q: 한글 폰트 깨짐
- 자동으로 Malgun Gothic 폰트 적용됨
- Windows 10 이상에서는 추가 설정 불필요

#### Q: PowerShell 스크립트 실행 오류
```powershell
# 관리자 권한 PowerShell에서
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Q: python 명령어를 찾을 수 없음
- Python 재설치 시 "Add Python to PATH" 체크
- 또는 `python3` 대신 `py` 명령어 사용

---

## 📂 프로젝트 구조

```
practice/
├── education.py           # 교육용 시각화
├── analysis.py           # 데이터 분석
├── activate_venv.sh     # macOS 자동 설정
├── requirements.txt     # 패키지 목록
├── README.md           # 사용 설명서
├── venv/              # 가상환경
├── visualizations/    # 교육용 그래프
├── data/             # 데이터 파일
│   ├── electricity_market.csv
│   ├── energy_demand.csv
│   └── renewable_policy.csv
├── models/           # 저장된 모델
│   ├── LSTM_model.h5
│   ├── GRU_model.h5
│   ├── RNN_model.h5
│   └── scaler.pkl
└── output/          # 분석 결과
```

---

## 📧 문의 및 지원

프로그램 사용 중 문제가 발생하면 다음 정보와 함께 문의하세요:
- OS 종류 및 버전
- Python 버전
- 오류 메시지 스크린샷