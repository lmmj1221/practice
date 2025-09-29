# 딥러닝 기반 정책 시계열 예측 시스템

## 프로젝트 구성

- `chap03_analysis.py`: 딥러닝 기반 정책 시계열 예측 분석 프로그램
- `education.py`: 딥러닝 개념 교육용 시각화 프로그램
- `analysis.py`: 실제 데이터 분석 및 모델 학습 프로그램

---

## 실행 방법

### 🪟 Windows

```cmd
# Command Prompt 또는 PowerShell에서
python chap03_analysis.py
python education.py
python analysis.py
```

### 🍎 macOS

```bash
# Terminal에서
python3 chap03_analysis.py
python3 education.py
python3 analysis.py
```

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
