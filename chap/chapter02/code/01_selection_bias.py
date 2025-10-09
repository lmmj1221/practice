"""
제2장: 선택편향 분석 실습
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def simulate_selection_bias(n=1000, seed=2025):
    """Generate selection bias simulation data"""
    np.random.seed(seed)
    
    # Generate covariates
    ability = np.random.normal(0, 1, n)
    education = np.random.uniform(6, 18, n)
    
    # Treatment probability (determined by ability and education)
    propensity_score = 1 / (1 + np.exp(-0.5 * ability - 0.2 * education + 2))
    treatment = np.random.binomial(1, propensity_score)
    
    # Potential outcomes
    y0 = 30 + 5 * ability + 2 * education + np.random.normal(0, 5, n)
    y1 = y0 + 10 + 2 * ability  # Treatment effect increases with ability
    
    # Observed outcome
    income = treatment * y1 + (1 - treatment) * y0
    
    return pd.DataFrame({
        'ability': ability,
        'education': education,
        'treatment': treatment,
        'income': income,
        'propensity_score': propensity_score,
        'true_effect': y1 - y0
    })

def main():
    # 데이터 생성
    df = simulate_selection_bias(n=2000)
    
    print("=== Selection Bias Empirical Analysis ===")
    print(f"Total sample size: {len(df)}")
    print(f"Treatment group ratio: {df['treatment'].mean()*100:.1f}%")
    
    # True average treatment effect
    true_ate = df['true_effect'].mean()
    print(f"\nTrue ATE: {true_ate:.2f}")
    
    # 1. Simple mean difference (biased estimate)
    naive_ate = df[df['treatment']==1]['income'].mean() - \
                df[df['treatment']==0]['income'].mean()
    print(f"Naive ATE: {naive_ate:.2f}")
    print(f"Selection bias: {naive_ate - true_ate:.2f}")
    
    # 2. Covariate-adjusted estimate (regression)
    X = df[['treatment', 'ability', 'education']]
    y = df['income']
    
    reg = LinearRegression()
    reg.fit(X, y)
    adjusted_ate = reg.coef_[0]  # treatment의 계수
    
    print(f"\nAdjusted ATE: {adjusted_ate:.2f}")
    print(f"Bias after adjustment: {adjusted_ate - true_ate:.2f}")
    
    # 3. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 3.1 Covariate distribution for treatment and control groups
    axes[0, 0].hist(df[df['treatment']==1]['ability'], alpha=0.5, 
                    label='Treated', bins=30, density=True)
    axes[0, 0].hist(df[df['treatment']==0]['ability'], alpha=0.5, 
                    label='Control', bins=30, density=True)
    axes[0, 0].set_xlabel('Ability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # 3.2 Propensity score distribution
    axes[0, 1].hist(df[df['treatment']==1]['propensity_score'], alpha=0.5, 
                    label='Treated', bins=30, density=True)
    axes[0, 1].hist(df[df['treatment']==0]['propensity_score'], alpha=0.5, 
                    label='Control', bins=30, density=True)
    axes[0, 1].set_xlabel('Propensity Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # 3.3 Relationship between ability and treatment effect
    axes[1, 0].scatter(df['ability'], df['true_effect'], alpha=0.3)
    axes[1, 0].set_xlabel('Ability')
    axes[1, 0].set_ylabel('True Treatment Effect')
    
    # 3.4 Comparison of estimation methods
    methods = ['True ATE', 'Naive', 'Adjusted']
    values = [true_ate, naive_ate, adjusted_ate]
    colors = ['green', 'red', 'blue']
    
    axes[1, 1].bar(methods, values, color=colors)
    axes[1, 1].axhline(y=true_ate, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel('Treatment Effect')
    
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('selection_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 데이터 저장
    df.to_csv('../../data/chapter02_data.csv', index=False)
    print("\nData saved: ../../data/chapter02_data.csv")

if __name__ == "__main__":
    main()