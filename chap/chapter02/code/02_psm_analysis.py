"""
제2장: 성향점수매칭(PSM) 분석 실습
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def calculate_smd(treated, control, var_name):
    """Calculate Standardized Mean Difference (SMD)"""
    mean_diff = treated[var_name].mean() - control[var_name].mean()
    pooled_std = np.sqrt((treated[var_name].var() + control[var_name].var()) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0

def perform_matching(df, caliper=0.1):
    """Perform 1:1 nearest neighbor matching"""
    treated = df[df['treatment'] == 1].copy()
    control = df[df['treatment'] == 0].copy()
    
    # NearestNeighbors를 사용한 매칭
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['ps']].values)
    
    matches = []
    for idx in treated.index:
        ps_treated = df.loc[idx, 'ps']
        distances, indices = nn.kneighbors([[ps_treated]])
        
        if distances[0][0] <= caliper:
            control_idx = control.iloc[indices[0][0]].name
            matches.append((idx, control_idx))
    
    return matches

def main():
    # Load data (generated from previous script)
    try:
        df = pd.read_csv('../../data/chapter02_data.csv')
    except:
        print("Data file not found. Please run 01_selection_bias.py first.")
        return
    
    print("=== Propensity Score Matching (PSM) Analysis ===")
    print(f"Total sample size: {len(df)}")
    
    # Create basic pension scenario
    df['age'] = 50 + np.random.exponential(10, len(df))  # Age distribution
    df['age'] = np.clip(df['age'], 50, 80)
    df['elderly'] = (df['age'] >= 65).astype(int)
    
    # Analyze only elderly
    elderly_df = df[df['elderly'] == 1].copy()
    print(f"Elderly sample size: {len(elderly_df)}")
    
    # Create region variable
    regions = ['Seoul', 'Gyeonggi', 'Busan', 'Daegu', 'Other']
    elderly_df['region'] = np.random.choice(regions, len(elderly_df), 
                                           p=[0.2, 0.25, 0.15, 0.1, 0.3])
    
    # 1. Estimate propensity scores
    X = pd.get_dummies(elderly_df[['age', 'education', 'ability', 'region']], 
                       columns=['region'], drop_first=True)
    y = elderly_df['treatment']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(max_iter=1000, random_state=2025)
    ps_model.fit(X_scaled, y)
    
    elderly_df['ps'] = ps_model.predict_proba(X_scaled)[:, 1]
    
    print(f"\nPropensity score range: [{elderly_df['ps'].min():.3f}, {elderly_df['ps'].max():.3f}]")
    
    # 2. Perform matching
    matches = perform_matching(elderly_df, caliper=0.1)
    print(f"Number of matched pairs: {len(matches)}")
    
    if len(matches) == 0:
        print("No matched pairs found. Please adjust the caliper.")
        return
    
    # 3. ATT (Average Treatment effect on the Treated) 계산
    treated_indices = [m[0] for m in matches]
    control_indices = [m[1] for m in matches]
    
    treated_outcomes = elderly_df.loc[treated_indices, 'income'].mean()
    control_outcomes = elderly_df.loc[control_indices, 'income'].mean()
    att = treated_outcomes - control_outcomes
    
    print(f"\n=== PSM Results ===")
    print(f"Basic Pension Effect (ATT): {att:.2f}")
    
    # 4. Balance test
    print("\n=== Balance Test (SMD) ===")
    variables = ['age', 'education', 'ability']
    
    matched_treated = elderly_df.loc[treated_indices]
    matched_control = elderly_df.loc[control_indices]
    
    for var in variables:
        smd_before = calculate_smd(
            elderly_df[elderly_df['treatment']==1],
            elderly_df[elderly_df['treatment']==0],
            var
        )
        smd_after = calculate_smd(matched_treated, matched_control, var)
        print(f"{var}: Before={smd_before:.3f}, After={smd_after:.3f}")
    
    # 5. 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 5.1 Propensity score distribution (before matching)
    axes[0, 0].hist(elderly_df[elderly_df['treatment']==1]['ps'], 
                    alpha=0.5, label='Treated', bins=20, density=True)
    axes[0, 0].hist(elderly_df[elderly_df['treatment']==0]['ps'], 
                    alpha=0.5, label='Control', bins=20, density=True)
    axes[0, 0].set_xlabel('Propensity Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # 5.2 Propensity score distribution (after matching)
    axes[0, 1].hist(matched_treated['ps'], 
                    alpha=0.5, label='Treated', bins=20, density=True)
    axes[0, 1].hist(matched_control['ps'], 
                    alpha=0.5, label='Control', bins=20, density=True)
    axes[0, 1].set_xlabel('Propensity Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # 5.3 Common support region
    axes[0, 2].scatter(elderly_df['ps'], elderly_df['treatment'] + 
                       np.random.normal(0, 0.02, len(elderly_df)), 
                       alpha=0.3, s=10)
    axes[0, 2].set_xlabel('Propensity Score')
    axes[0, 2].set_ylabel('Treatment Status')
    axes[0, 2].set_yticks([0, 1])
    axes[0, 2].set_yticklabels(['Control', 'Treated'])
    
    # 5.4 Matching quality
    balance_data = pd.DataFrame({
        'Variable': variables * 2,
        'SMD': [calculate_smd(elderly_df[elderly_df['treatment']==1],
                              elderly_df[elderly_df['treatment']==0], var) 
                for var in variables] +
               [calculate_smd(matched_treated, matched_control, var) 
                for var in variables],
        'Type': ['Before'] * len(variables) + ['After'] * len(variables)
    })
    
    x = np.arange(len(variables))
    width = 0.35
    
    before_smd = balance_data[balance_data['Type']=='Before']['SMD'].values
    after_smd = balance_data[balance_data['Type']=='After']['SMD'].values
    
    axes[1, 0].bar(x - width/2, np.abs(before_smd), width, label='Before Matching')
    axes[1, 0].bar(x + width/2, np.abs(after_smd), width, label='After Matching')
    axes[1, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Balance Threshold (0.1)')
    axes[1, 0].set_xlabel('Variable')
    axes[1, 0].set_ylabel('|SMD|')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(variables)
    axes[1, 0].legend()
    
    # 5.5 Treatment effect distribution
    # Calculate effect for each matched pair
    pair_effects = []
    for t_idx, c_idx in matches:
        effect = elderly_df.loc[t_idx, 'income'] - elderly_df.loc[c_idx, 'income']
        pair_effects.append(effect)
    
    axes[1, 1].hist(pair_effects, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=att, color='red', linestyle='--', linewidth=2, 
                       label=f'ATT: {att:.2f}')
    axes[1, 1].set_xlabel('Individual Treatment Effect')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # 5.6 Results summary
    summary_text = f"""
    PSM Analysis Results
    ==================
    Total elderly sample: {len(elderly_df)}
    Matched pairs: {len(matches)}
    Matching rate: {len(matches)/elderly_df['treatment'].sum()*100:.1f}%
    
    Treatment effect (ATT): {att:.2f}
    Standard error: {np.std(pair_effects)/np.sqrt(len(pair_effects)):.2f}
    95% CI: [{att - 1.96*np.std(pair_effects)/np.sqrt(len(pair_effects)):.2f}, 
             {att + 1.96*np.std(pair_effects)/np.sqrt(len(pair_effects)):.2f}]
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('psm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()