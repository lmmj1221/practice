# 제3장 데이터 코드북

## 1. 정책 시계열 데이터 구조

### 1.1 경제 지표 데이터
```
columns:
- date: 날짜 (YYYY-MM-DD)
- gdp_growth: GDP 성장률 (%)
- unemployment_rate: 실업률 (%)
- inflation_rate: 인플레이션율 (%)
- interest_rate: 기준금리 (%)
- policy_intervention: 정책 개입 여부 (0/1)
```

### 1.2 에너지 수요 데이터
```
columns:
- timestamp: 시간 (YYYY-MM-DD HH:MM:SS)
- energy_demand: 에너지 수요량 (MWh)
- temperature: 온도 (°C)
- humidity: 습도 (%)
- holiday: 공휴일 여부 (0/1)
- hour_of_day: 시간대 (0-23)
- day_of_week: 요일 (0-6)
```

## 2. 모델 입력 변수

### 2.1 정적 공변량 (Static Covariates)
- region_id: 지역 식별자
- sector_type: 산업 분류
- population_size: 인구 규모

### 2.2 시간 의존 공변량 (Time-varying Covariates)
- known_inputs: 미래 알려진 변수 (날짜, 공휴일)
- unknown_inputs: 미래 모르는 변수 (날씨, 경제지표)

## 3. 타겟 변수
- target: 예측 대상 (GDP, 에너지 수요 등)
- target_normalized: 정규화된 타겟 값