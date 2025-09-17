"""
제5장 실습용 정책 데이터 생성 스크립트
- 정부 예산 배분 데이터
- 경제 지표 시계열 데이터
- 정책 문서 텍스트 데이터
- 도시 개발 이미지 데이터 (시뮬레이션)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import json

# 랜덤 시드 설정
np.random.seed(42)

def generate_budget_data(n_samples=1000):
    """정부 예산 배분 데이터 생성"""
    
    # 부처별 예산 데이터
    departments = ['교육부', '복지부', '국방부', '환경부', '산업부', '국토부', '과기부', '문체부']
    years = list(range(2020, 2026))
    
    data = []
    for _ in range(n_samples):
        year = np.random.choice(years)
        dept = np.random.choice(departments)
        
        # 기본 예산 (단위: 억원)
        base_budget = np.random.uniform(1000, 50000)
        
        # 특성 변수들
        gdp_growth = np.random.uniform(-2, 5)  # GDP 성장률
        unemployment = np.random.uniform(2, 8)  # 실업률
        inflation = np.random.uniform(-1, 4)    # 인플레이션
        dept_priority = np.random.uniform(0, 1)  # 부처 우선순위
        
        # 정책 효과 (목표 변수)
        policy_effect = (
            0.3 * np.log(base_budget + 1) +
            0.2 * gdp_growth +
            -0.15 * unemployment +
            -0.1 * inflation +
            0.25 * dept_priority +
            np.random.normal(0, 0.1)
        )
        
        data.append({
            'year': year,
            'department': dept,
            'budget': base_budget,
            'gdp_growth': gdp_growth,
            'unemployment': unemployment,
            'inflation': inflation,
            'dept_priority': dept_priority,
            'policy_effect': policy_effect
        })
    
    df = pd.DataFrame(data)
    return df

def generate_economic_timeseries(days=365*3):
    """경제 지표 시계열 데이터 생성"""
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # 경제 지표들
    gdp_trend = np.cumsum(np.random.randn(days) * 0.1) + 100
    unemployment_trend = 4 + np.sin(np.arange(days) * 2 * np.pi / 365) + np.random.randn(days) * 0.2
    interest_rate = 2 + np.cumsum(np.random.randn(days) * 0.01)
    exchange_rate = 1300 + np.cumsum(np.random.randn(days) * 5)
    
    # 정책 개입 효과 시뮬레이션
    policy_interventions = np.zeros(days)
    intervention_dates = [100, 250, 400, 550, 700]
    for date in intervention_dates:
        if date < days:
            policy_interventions[date:date+30] = np.random.uniform(0.5, 1.5)
    
    df = pd.DataFrame({
        'date': dates,
        'gdp_index': gdp_trend,
        'unemployment_rate': unemployment_trend,
        'interest_rate': interest_rate,
        'exchange_rate': exchange_rate,
        'policy_intervention': policy_interventions
    })
    
    return df

def generate_policy_texts(n_samples=500):
    """정책 문서 텍스트 데이터 생성"""
    
    policy_types = ['경제', '교육', '복지', '환경', '산업']
    sentiments = ['긍정', '중립', '부정']
    
    templates = {
        '경제': [
            "경제 성장률 {rate}% 달성을 위한 {action} 정책 시행",
            "중소기업 지원을 통한 {effect} 효과 기대",
            "금융 시장 안정화를 위한 {measure} 도입"
        ],
        '교육': [
            "교육 격차 해소를 위한 {program} 프로그램 확대",
            "디지털 교육 인프라 {investment} 투자 계획",
            "창의적 인재 양성을 위한 {curriculum} 개편"
        ],
        '복지': [
            "취약계층 지원 {amount} 확대 방안",
            "고령화 대응 {service} 서비스 강화",
            "사회안전망 {improvement} 개선 추진"
        ],
        '환경': [
            "탄소중립 달성을 위한 {target} 목표 설정",
            "재생에너지 {percentage}% 확대 계획",
            "친환경 {industry} 산업 육성 정책"
        ],
        '산업': [
            "첨단산업 {technology} 기술 개발 지원",
            "제조업 경쟁력 {enhancement} 강화 방안",
            "스타트업 {ecosystem} 생태계 활성화"
        ]
    }
    
    data = []
    for _ in range(n_samples):
        policy_type = np.random.choice(policy_types)
        sentiment = np.random.choice(sentiments)
        template = np.random.choice(templates[policy_type])
        
        # 템플릿 채우기
        text = template.format(
            rate=np.random.uniform(2, 5),
            action=np.random.choice(['혁신', '확대', '강화']),
            effect=np.random.choice(['고용창출', '생산성향상', '경쟁력강화']),
            measure=np.random.choice(['규제완화', '세제혜택', '금융지원']),
            program=np.random.choice(['스마트', '미래', '혁신']),
            investment=np.random.choice(['대규모', '지속적', '전략적']),
            curriculum=np.random.choice(['전면', '단계적', '혁신적']),
            amount=np.random.choice(['대폭', '점진적', '선별적']),
            service=np.random.choice(['맞춤형', '통합', '디지털']),
            improvement=np.random.choice(['전면적', '단계별', '지속적']),
            target=np.random.choice(['도전적', '현실적', '단계별']),
            percentage=np.random.randint(20, 50),
            industry=np.random.choice(['모빌리티', '에너지', '순환경제']),
            technology=np.random.choice(['AI', '바이오', '반도체']),
            enhancement=np.random.choice(['획기적', '지속적', '전략적']),
            ecosystem=np.random.choice(['혁신', '창업', '투자'])
        )
        
        # 정책 효과 점수 (목표 변수)
        effect_score = np.random.uniform(0, 1)
        if sentiment == '긍정':
            effect_score += 0.3
        elif sentiment == '부정':
            effect_score -= 0.3
            
        data.append({
            'policy_type': policy_type,
            'text': text,
            'sentiment': sentiment,
            'effect_score': np.clip(effect_score, 0, 1)
        })
    
    df = pd.DataFrame(data)
    return df

def generate_urban_development_data(n_samples=200):
    """도시 개발 데이터 생성 (이미지 메타데이터)"""
    
    regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종']
    dev_types = ['주거', '상업', '산업', '녹지', '복합']
    
    data = []
    for _ in range(n_samples):
        region = np.random.choice(regions)
        dev_type = np.random.choice(dev_types)
        
        # 개발 특성
        area_size = np.random.uniform(1000, 100000)  # 평방미터
        green_ratio = np.random.uniform(0.1, 0.5)     # 녹지 비율
        building_height = np.random.uniform(5, 50)    # 평균 층수
        population_density = np.random.uniform(100, 10000)  # 인구밀도
        
        # 이미지 특성 (시뮬레이션)
        avg_brightness = np.random.uniform(100, 200)
        edge_density = np.random.uniform(0.1, 0.9)
        color_diversity = np.random.uniform(0.2, 0.8)
        
        # 개발 효과 점수
        dev_score = (
            0.2 * np.log(area_size + 1) / 10 +
            0.3 * green_ratio +
            0.1 * (1 / (1 + np.exp(-building_height/20))) +
            -0.2 * np.log(population_density + 1) / 10 +
            0.1 * color_diversity +
            np.random.normal(0, 0.05)
        )
        
        data.append({
            'region': region,
            'dev_type': dev_type,
            'area_size': area_size,
            'green_ratio': green_ratio,
            'building_height': building_height,
            'population_density': population_density,
            'avg_brightness': avg_brightness,
            'edge_density': edge_density,
            'color_diversity': color_diversity,
            'development_score': np.clip(dev_score, 0, 1)
        })
    
    df = pd.DataFrame(data)
    return df

def save_datasets():
    """모든 데이터셋 저장"""
    
    # 디렉토리 생성
    os.makedirs('data', exist_ok=True)
    
    # 1. 예산 데이터
    budget_df = generate_budget_data()
    budget_df.to_csv('data/government_budget.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 정부 예산 데이터 저장 완료: {budget_df.shape}")
    
    # 2. 경제 시계열 데이터
    economic_df = generate_economic_timeseries()
    economic_df.to_csv('data/economic_indicators.csv', index=False)
    print(f"✅ 경제 지표 데이터 저장 완료: {economic_df.shape}")
    
    # 3. 정책 텍스트 데이터
    policy_df = generate_policy_texts()
    policy_df.to_csv('data/policy_documents.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 정책 문서 데이터 저장 완료: {policy_df.shape}")
    
    # 4. 도시 개발 데이터
    urban_df = generate_urban_development_data()
    urban_df.to_csv('data/urban_development.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 도시 개발 데이터 저장 완료: {urban_df.shape}")
    
    # 메타데이터 저장
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'datasets': {
            'government_budget': {'rows': len(budget_df), 'cols': len(budget_df.columns)},
            'economic_indicators': {'rows': len(economic_df), 'cols': len(economic_df.columns)},
            'policy_documents': {'rows': len(policy_df), 'cols': len(policy_df.columns)},
            'urban_development': {'rows': len(urban_df), 'cols': len(urban_df.columns)}
        }
    }
    
    with open('data/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("\n📊 모든 데이터셋 생성 완료!")
    return budget_df, economic_df, policy_df, urban_df

if __name__ == "__main__":
    save_datasets()