"""
제2장: 정책분석을 위한 인과추론 기초와 머신러닝 결합
1. 선택편향 분석 및 기초연금 정책 평가

실제 데이터: 한국 기초연금 통계 (2024-2025)
- 수급자: 736만명
- 월 지급액: 2024년 334,810원 → 2025년 342,510원
- 소득 기준: 단독가구 228만원, 부부가구 364.8만원
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정 (한글 폰트 깨짐 방지)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 난수 시드 고정 (재현성 보장)
np.random.seed(2025)

def generate_pension_data(n=1000):
    """
    한국 기초연금 정책 효과 분석을 위한 데이터 생성
    실제 통계를 반영한 현실적 시뮬레이션
    
    Returns:
        DataFrame: 분석용 데이터
    """
    # 기본 특성 생성
    age = np.random.uniform(65, 85, n)
    education = np.random.normal(9, 3, n).clip(0, 18)  # 교육연수
    ability = np.random.normal(0, 1, n)
    
    # 소득인정액 생성 (실제 기준 반영: 228만원 기준)
    base_income = 150 + education * 5 + ability * 20 + np.random.normal(0, 30, n)
    
    # 기초연금 수급 여부 (소득인정액 228만원 이하)
    elderly = (age >= 65).astype(int)
    eligible = (base_income < 228) & (elderly == 1)
    
    # 처치 확률 (성향점수) - 소득이 낮을수록 수급 확률 높음
    propensity = 1 / (1 + np.exp(0.02 * (base_income - 228)))
    treatment = np.where(eligible, np.random.binomial(1, propensity), 0)
    
    # 실제 기초연금 효과 (월 34.3만원 지급)
    true_effect = 34.3  # 2025년 기준 월 지급액
    
    # 관찰된 소득 (기초연금 포함)
    income = base_income + treatment * true_effect + np.random.normal(0, 10, n)
    
    df = pd.DataFrame({
        'age': age,
        'education': education,
        'ability': ability,
        'base_income': base_income,
        'elderly': elderly,
        'treatment': treatment,
        'income': income,
        'propensity_score': propensity
    })
    
    return df

def analyze_selection_bias(df):
    """선택편향 분석"""
    
    # 노인 대상자만 분석
    elderly_df = df[df['elderly'] == 1].copy()
    
    # 1. 단순 평균 차이
    naive_ate = elderly_df[elderly_df['treatment']==1]['income'].mean() - \
                elderly_df[elderly_df['treatment']==0]['income'].mean()
    
    # 2. 공변량 조정 회귀분석
    X = elderly_df[['treatment', 'ability', 'education']]
    y = elderly_df['income']
    reg = LinearRegression().fit(X, y)
    adjusted_ate = reg.coef_[0]
    
    # 3. 진짜 처치효과 계산 (기초연금 월 지급액)
    true_ate = 34.3
    
    results = {
        '진짜 ATE (기초연금 월액)': true_ate,
        '단순 평균 차이': naive_ate,
        '편향 (단순)': naive_ate - true_ate,
        '공변량 조정': adjusted_ate,
        '편향 (조정)': adjusted_ate - true_ate
    }
    
    return results, elderly_df

def visualize_selection_bias(elderly_df, results):
    """선택편향 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 처치군과 대조군의 소득 분포
    treated = elderly_df[elderly_df['treatment'] == 1]
    control = elderly_df[elderly_df['treatment'] == 0]
    
    axes[0, 0].hist(control['base_income'], bins=20, alpha=0.5, label='Control', color='blue')
    axes[0, 0].hist(treated['base_income'], bins=20, alpha=0.5, label='Treated', color='red')
    axes[0, 0].axvline(228, color='green', linestyle='--', label='Eligibility Threshold (228)')
    axes[0, 0].set_xlabel('Base Income (10,000 KRW)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Income Distribution by Treatment Status')
    axes[0, 0].legend()
    
    # 2. 성향점수 분포
    axes[0, 1].hist(control['propensity_score'], bins=20, alpha=0.5, label='Control', color='blue')
    axes[0, 1].hist(treated['propensity_score'], bins=20, alpha=0.5, label='Treated', color='red')
    axes[0, 1].set_xlabel('Propensity Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Propensity Score Distribution')
    axes[0, 1].legend()
    
    # 3. 교육수준과 처치효과
    axes[1, 0].scatter(elderly_df['education'], elderly_df['income'], 
                      c=elderly_df['treatment'], cmap='coolwarm', alpha=0.6)
    axes[1, 0].set_xlabel('Education (years)')
    axes[1, 0].set_ylabel('Income (10,000 KRW)')
    axes[1, 0].set_title('Education vs Income by Treatment')
    axes[1, 0].colorbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    
    # 4. 추정 방법별 비교
    methods = ['True ATE\n(34.3)', 'Naive\nDifference', 'Covariate\nAdjusted']
    values = [34.3, results['단순 평균 차이'], results['공변량 조정']]
    colors = ['green', 'red', 'blue']
    
    bars = axes[1, 1].bar(methods, values, color=colors, alpha=0.7)
    axes[1, 1].axhline(34.3, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel('Treatment Effect (10,000 KRW)')
    axes[1, 1].set_title('Comparison of Estimation Methods')
    axes[1, 1].set_ylim([0, max(values) * 1.2])
    
    # 값 표시
    for bar, val in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom')
    
    plt.suptitle('Selection Bias Analysis: Korean Basic Pension (2025)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('evaluation/outputs/selection_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """메인 실행 함수"""
    
    print("="*60)
    print("제2장: 인과추론 기초 - 한국 기초연금 정책 분석")
    print("="*60)
    
    # 데이터 생성
    print("\n1. 데이터 생성 중...")
    df = generate_pension_data(n=1000)
    
    # 데이터 저장
    df.to_csv('evaluation/data/chapter02/pension_data.csv', index=False)
    print(f"   - 생성된 샘플 수: {len(df)}")
    print(f"   - 노인 대상자: {df['elderly'].sum()}명")
    print(f"   - 기초연금 수급자: {df['treatment'].sum()}명")
    
    # 선택편향 분석
    print("\n2. 선택편향 분석 중...")
    results, elderly_df = analyze_selection_bias(df)
    
    print("\n[선택편향 분석 결과]")
    print("-" * 40)
    for key, value in results.items():
        print(f"{key:25s}: {value:8.2f}만원")
    
    # 시각화
    print("\n3. 시각화 생성 중...")
    fig = visualize_selection_bias(elderly_df, results)
    
    # 결과 요약
    print("\n[분석 요약]")
    print("-" * 40)
    print(f"• 2025년 기초연금 월 지급액: 34.3만원")
    print(f"• 단순 비교 편향: {results['편향 (단순)']:.2f}만원")
    print(f"• 공변량 조정 후 편향: {results['편향 (조정)']:.2f}만원")
    print(f"• 편향 감소율: {abs(results['편향 (조정)'])/abs(results['편향 (단순)'])*100:.1f}%")
    
    print("\n분석 완료! 결과가 evaluation/outputs/에 저장되었습니다.")
    
    return df, results

if __name__ == "__main__":
    df, results = main()