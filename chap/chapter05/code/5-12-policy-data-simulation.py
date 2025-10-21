"""
COVID-19 경제 부양 정책 효과 시뮬레이션 데이터 생성
- 고용률, GDP, 소비지수 등에 미치는 영향 분석
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_policy_effect_data():
    """
    정책 효과 시계열 데이터 생성
    2020년 1월부터 2023년 12월까지 (48개월)
    """

    # 기간 설정
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n_months = len(dates)

    # 1. 정책 변수들 (Policy Variables)
    # 재난지원금 규모 (조 원)
    emergency_relief = np.zeros(n_months)
    emergency_relief[2:4] = [14.3, 12.2]  # 2020년 3-4월 1차
    emergency_relief[8:10] = [7.8, 9.6]   # 2020년 9-10월 2차
    emergency_relief[14:16] = [11.0, 9.8] # 2021년 3-4월 3차
    emergency_relief[20:22] = [14.0, 13.5] # 2021년 9-10월 4차

    # 고용유지 지원금 (조 원)
    employment_support = np.zeros(n_months)
    employment_support[2:24] = np.random.uniform(2.5, 4.5, 22)  # 2020.3 - 2021.12
    employment_support[24:36] = np.random.uniform(1.5, 2.5, 12) # 2022년

    # 소상공인 지원금 (조 원)
    small_business_support = np.zeros(n_months)
    small_business_support[3:30] = np.random.uniform(1.0, 3.0, 27)

    # 금리 인하 효과 (기준금리 변화)
    interest_rate = np.ones(n_months) * 1.25  # 시작 금리
    interest_rate[2:12] = 0.5   # 2020년 금리 인하
    interest_rate[12:24] = 0.5  # 2021년 저금리 유지
    interest_rate[24:36] = np.linspace(0.5, 2.0, 12)  # 2022년 금리 인상
    interest_rate[36:] = np.linspace(2.0, 3.5, 12)    # 2023년 금리 인상

    # 2. 외부 요인 (External Factors)
    # COVID-19 심각도 (0-100)
    covid_severity = np.zeros(n_months)
    covid_severity[2:6] = [30, 70, 90, 60]    # 1차 대유행
    covid_severity[6:12] = [40, 35, 45, 80, 85, 70]  # 2차 대유행
    covid_severity[12:18] = [50, 45, 40, 60, 65, 55]  # 3차 대유행
    covid_severity[18:24] = [40, 35, 30, 45, 50, 40]  # 델타 변이
    covid_severity[24:30] = [35, 30, 25, 20, 15, 10]  # 오미크론 및 완화
    covid_severity[30:] = np.random.uniform(5, 15, n_months-30)  # 엔데믹

    # 글로벌 경제 지표 (정규화된 값)
    global_economy = np.sin(np.linspace(0, 4*np.pi, n_months)) * 0.3 + \
                     np.random.normal(0, 0.1, n_months)

    # 3. 정책 효과 계산 (시차 반영)
    # 고용률 (%)
    base_employment = 60.0
    employment_rate = np.zeros(n_months)

    for i in range(n_months):
        # 기본 고용률
        employment_rate[i] = base_employment

        # 재난지원금 효과 (1-2개월 시차)
        if i >= 1:
            employment_rate[i] += emergency_relief[i-1] * 0.15
        if i >= 2:
            employment_rate[i] += emergency_relief[i-2] * 0.10

        # 고용유지 지원금 효과 (즉시)
        employment_rate[i] += employment_support[i] * 0.8

        # 소상공인 지원금 효과 (1개월 시차)
        if i >= 1:
            employment_rate[i] += small_business_support[i-1] * 0.5

        # 금리 효과 (2-3개월 시차)
        if i >= 2:
            employment_rate[i] += (1.25 - interest_rate[i-2]) * 2.0

        # COVID-19 부정적 효과
        employment_rate[i] -= covid_severity[i] * 0.08

        # 글로벌 경제 효과
        employment_rate[i] += global_economy[i] * 2.0

        # 노이즈 추가
        employment_rate[i] += np.random.normal(0, 0.5)

    # 4. GDP 성장률 (%)
    gdp_growth = np.zeros(n_months)
    for i in range(n_months):
        gdp_growth[i] = 2.0  # 기본 성장률

        # 정책 효과들
        if i >= 1:
            gdp_growth[i] += emergency_relief[i-1] * 0.08
        gdp_growth[i] += employment_support[i] * 0.05
        if i >= 1:
            gdp_growth[i] += small_business_support[i-1] * 0.03
        if i >= 3:
            gdp_growth[i] += (1.25 - interest_rate[i-3]) * 1.5

        # COVID-19 효과
        gdp_growth[i] -= covid_severity[i] * 0.05

        # 글로벌 경제
        gdp_growth[i] += global_economy[i] * 3.0

        # 노이즈
        gdp_growth[i] += np.random.normal(0, 0.3)

    # 5. 소비자 신뢰지수 (0-100)
    consumer_confidence = np.zeros(n_months)
    for i in range(n_months):
        consumer_confidence[i] = 50  # 기본값

        # 재난지원금 즉시 효과
        consumer_confidence[i] += emergency_relief[i] * 1.5

        # COVID-19 부정적 효과
        consumer_confidence[i] -= covid_severity[i] * 0.3

        # 고용률 영향 (1개월 시차)
        if i >= 1:
            consumer_confidence[i] += (employment_rate[i-1] - 60) * 2.0

        # 노이즈
        consumer_confidence[i] += np.random.normal(0, 2.0)
        consumer_confidence[i] = np.clip(consumer_confidence[i], 0, 100)

    # 6. 물가상승률 (%)
    inflation = np.zeros(n_months)
    for i in range(n_months):
        inflation[i] = 2.0  # 목표 물가상승률

        # 재난지원금 효과 (2-3개월 시차)
        if i >= 2:
            inflation[i] += emergency_relief[i-2] * 0.05
        if i >= 3:
            inflation[i] += emergency_relief[i-3] * 0.03

        # 금리 효과 (3-4개월 시차)
        if i >= 3:
            inflation[i] += (1.25 - interest_rate[i-3]) * 0.8

        # 글로벌 요인
        inflation[i] += global_economy[i] * 1.0

        # 노이즈
        inflation[i] += np.random.normal(0, 0.2)

    # 7. 실업급여 신청건수 (천 건)
    unemployment_claims = np.zeros(n_months)
    for i in range(n_months):
        # 고용률과 반비례
        unemployment_claims[i] = (65 - employment_rate[i]) * 50

        # COVID-19 효과
        unemployment_claims[i] += covid_severity[i] * 3.0

        # 노이즈
        unemployment_claims[i] += np.random.normal(0, 10)
        unemployment_claims[i] = max(0, unemployment_claims[i])

    # 데이터프레임 생성
    df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'emergency_relief': emergency_relief,
        'employment_support': employment_support,
        'small_business_support': small_business_support,
        'interest_rate': interest_rate,
        'covid_severity': covid_severity,
        'global_economy': global_economy,
        'employment_rate': employment_rate,
        'gdp_growth': gdp_growth,
        'consumer_confidence': consumer_confidence,
        'inflation': inflation,
        'unemployment_claims': unemployment_claims
    })

    # 추가 파생 변수
    df['total_stimulus'] = df['emergency_relief'] + df['employment_support'] + df['small_business_support']
    df['employment_change'] = df['employment_rate'].diff()
    df['gdp_change'] = df['gdp_growth'].diff()

    # 이동평균 추가 (트렌드 파악용)
    df['employment_ma3'] = df['employment_rate'].rolling(window=3).mean()
    df['gdp_ma3'] = df['gdp_growth'].rolling(window=3).mean()
    df['confidence_ma3'] = df['consumer_confidence'].rolling(window=3).mean()

    return df

def save_policy_data():
    """데이터를 CSV 파일로 저장"""
    # data 폴더 경로
    data_dir = 'c:/practice/chap/chapter05/data'
    os.makedirs(data_dir, exist_ok=True)

    # 데이터 생성
    df = generate_policy_effect_data()

    # 저장
    file_path = os.path.join(data_dir, 'covid_policy_effects.csv')
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"데이터가 저장되었습니다: {file_path}")

    # 데이터 요약 정보 출력
    print("\n=== 데이터 요약 ===")
    print(f"기간: {df['date'].min().strftime('%Y-%m')} ~ {df['date'].max().strftime('%Y-%m')}")
    print(f"총 {len(df)}개월 데이터")
    print("\n주요 변수:")
    print("- 정책 변수: 재난지원금, 고용유지지원금, 소상공인지원금, 기준금리")
    print("- 외부 요인: COVID-19 심각도, 글로벌 경제지표")
    print("- 결과 변수: 고용률, GDP 성장률, 소비자신뢰지수, 물가상승률, 실업급여신청")

    print("\n=== 기초 통계 ===")
    print(df[['employment_rate', 'gdp_growth', 'consumer_confidence', 'inflation']].describe())

    return df

if __name__ == "__main__":
    df = save_policy_data()