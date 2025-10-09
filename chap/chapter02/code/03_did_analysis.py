"""
Chapter 2: Difference-in-Differences (DID) Analysis
COVID-19 Emergency Relief Fund Effect Analysis
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def generate_panel_data(n_units=150, n_periods=12, seed=2025):
    """Generate COVID-19 emergency relief fund panel data"""
    np.random.seed(seed)
    
    data = []
    treatment_period = 5  # Payment in June
    
    for i in range(n_units):
        # Regional characteristics
        region_type = np.random.choice(['Seoul', 'Gyeonggi', 'Busan', 'Daegu', 'Other'])
        urban = 1 if region_type in ['Seoul', 'Gyeonggi'] else 0
        
        # Treatment assignment (Seoul, Gyeonggi receive early payment)
        treated = 1 if region_type in ['Seoul', 'Gyeonggi'] else 0
        
        # Regional fixed effect
        region_fe = np.random.normal(100, 20)
        
        for t in range(n_periods):
            # Time trend
            time_trend = 2 * t
            
            # Seasonal effect
            seasonal = 10 * np.sin(2 * np.pi * t / 12)
            
            # COVID-19 shock (from March)
            covid_shock = -30 if t >= 2 else 0
            
            # Treatment effect (emergency relief fund)
            post = 1 if t >= treatment_period else 0
            treatment_effect = 25 if (treated == 1 and post == 1) else 0
            
            # Consumption expenditure (outcome variable)
            consumption = (region_fe + time_trend + seasonal + 
                         covid_shock + treatment_effect + 
                         np.random.normal(0, 10))
            
            data.append({
                'unit_id': i,
                'time': t,
                'month': f'2020-{t+1:02d}',
                'region': region_type,
                'urban': urban,
                'treated': treated,
                'post': post,
                'treatment': treated * post,
                'consumption': consumption
            })
    
    return pd.DataFrame(data)

def run_event_study(panel_df):
    """Event Study Analysis"""
    # Estimate effects at each time point before/after treatment
    event_dummies = []
    for t in range(12):
        if t != 4:  # Using pre-treatment period as baseline
            panel_df[f'treat_time_{t}'] = \
                (panel_df['time'] == t).astype(int) * panel_df['treated']
            event_dummies.append(f'treat_time_{t}')
    
    event_formula = 'consumption ~ ' + ' + '.join(event_dummies) + ' + C(unit_id) + C(time)'
    event_model = smf.ols(event_formula, data=panel_df)
    event_results = event_model.fit()
    
    # Extract coefficients
    event_coefs = []
    event_ses = []
    periods = []
    
    for t in range(12):
        if t != 4:
            coef_name = f'treat_time_{t}'
            if coef_name in event_results.params:
                event_coefs.append(event_results.params[coef_name])
                event_ses.append(event_results.bse[coef_name])
                periods.append(t - 4)
    
    return periods, event_coefs, event_ses

def main():
    print("=== COVID-19 Emergency Relief Fund DID Analysis ===\n")
    
    # 1. Generate data
    panel_df = generate_panel_data(n_units=150, n_periods=12)
    
    print(f"Panel data structure:")
    print(f"- Total observations: {len(panel_df)}")
    print(f"- Number of regions: {panel_df['unit_id'].nunique()}")
    print(f"- Time periods: {panel_df['time'].nunique()} months")
    print(f"- Treatment group ratio: {panel_df.groupby('unit_id')['treated'].first().mean()*100:.1f}%")
    
    # 2. Basic DID analysis
    print("\n=== Basic DID Analysis ===")
    did_model = smf.ols('consumption ~ treated + post + treatment', data=panel_df)
    did_results = did_model.fit(cov_type='cluster', cov_kwds={'groups': panel_df['unit_id']})
    
    print(did_results.summary().tables[1])
    
    did_effect = did_results.params['treatment']
    did_se = did_results.bse['treatment']
    
    print(f"\nEmergency relief fund effect: {did_effect:.2f} (SE: {did_se:.2f})")
    print(f"95% confidence interval: [{did_results.conf_int().loc['treatment', 0]:.2f}, "
          f"{did_results.conf_int().loc['treatment', 1]:.2f}]")
    print(f"t-statistic: {did_results.tvalues['treatment']:.2f}, "
          f"p-value: {did_results.pvalues['treatment']:.4f}")
    
    # 3. Two-way fixed effects model
    print("\n=== Two-way Fixed Effects Model ===")
    panel_df_indexed = panel_df.set_index(['unit_id', 'time'])
    
    fe_model = PanelOLS(panel_df_indexed['consumption'], 
                         panel_df_indexed[['treatment']],
                         entity_effects=True,
                         time_effects=True)
    fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)
    
    print(f"Fixed effects model treatment effect: {fe_results.params['treatment']:.2f}")
    
    # 4. Event Study (parallel trends test)
    periods, event_coefs, event_ses = run_event_study(panel_df)
    
    # 5. Bootstrap confidence interval
    print("\n=== Bootstrap Analysis ===")
    n_bootstrap = 1000
    bootstrap_effects = []
    
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(panel_df['unit_id'].unique(), 
                                     size=len(panel_df['unit_id'].unique()), 
                                     replace=True)
        sample_df = pd.concat([panel_df[panel_df['unit_id']==idx] for idx in sample_idx])
        
        try:
            boot_model = smf.ols('consumption ~ treated + post + treatment', data=sample_df)
            boot_results = boot_model.fit()
            bootstrap_effects.append(boot_results.params['treatment'])
        except:
            continue
    
    ci_lower = np.percentile(bootstrap_effects, 2.5)
    ci_upper = np.percentile(bootstrap_effects, 97.5)
    
    print(f"Bootstrap 95% confidence interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # 6. Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 6.1 Time trends for treated and control groups
    treated_mean = panel_df[panel_df['treated']==1].groupby('time')['consumption'].mean()
    control_mean = panel_df[panel_df['treated']==0].groupby('time')['consumption'].mean()
    
    axes[0, 0].plot(treated_mean.index, treated_mean.values, 'o-', 
                    label='Treated (Seoul/Gyeonggi)', linewidth=2)
    axes[0, 0].plot(control_mean.index, control_mean.values, 's-', 
                    label='Control (Other regions)', linewidth=2)
    axes[0, 0].axvline(x=4.5, color='red', linestyle='--', alpha=0.5, 
                      label='Relief fund payment')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Average Consumption')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 6.2 DID visualization
    pre_post_data = panel_df.groupby(['treated', 'post'])['consumption'].mean().unstack()
    x = [0, 1]
    
    axes[0, 1].plot(x, pre_post_data.loc[1], 'o-', label='Treated', 
                    linewidth=2, markersize=8)
    axes[0, 1].plot(x, pre_post_data.loc[0], 's-', label='Control', 
                    linewidth=2, markersize=8)
    
    # Counterfactual treated (parallel trends assumption)
    counterfactual = pre_post_data.loc[0] + \
                     (pre_post_data.loc[1].iloc[0] - pre_post_data.loc[0].iloc[0])
    axes[0, 1].plot(x, counterfactual, ':', color='gray', alpha=0.7, 
                    label='Counterfactual Treated')
    
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Before', 'After'])
    axes[0, 1].set_ylabel('Average Consumption')
    axes[0, 1].legend()
    
    # DID effect arrow
    y1 = pre_post_data.loc[1].iloc[1]
    y2 = counterfactual.iloc[1]
    axes[0, 1].annotate('', xy=(1, y1-2), xytext=(1, y2+2),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    axes[0, 1].text(1.05, (y1 + y2)/2, f'DID = {did_effect:.1f}', 
                    fontsize=10, color='red')
    
    # 6.3 Event Study
    if len(periods) > 0:
        axes[0, 2].errorbar(periods, event_coefs, yerr=1.96*np.array(event_ses), 
                           fmt='o', capsize=5, capthick=2)
        axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 2].set_xlabel('Time Relative to Relief Fund Payment')
        axes[0, 2].set_ylabel('Treatment Effect')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 6.4 Bootstrap distribution
    axes[1, 0].hist(bootstrap_effects, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=did_effect, color='red', linestyle='--', 
                       linewidth=2, label=f'DID Estimate: {did_effect:.2f}')
    axes[1, 0].axvline(x=ci_lower, color='blue', linestyle=':', 
                       label=f'95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]')
    axes[1, 0].axvline(x=ci_upper, color='blue', linestyle=':')
    axes[1, 0].set_xlabel('Treatment Effect')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 6.5 Regional effects
    region_effects = panel_df[panel_df['post']==1].groupby('region')['consumption'].mean() - \
                     panel_df[panel_df['post']==0].groupby('region')['consumption'].mean()
    
    axes[1, 1].bar(region_effects.index, region_effects.values, 
                   color=['red' if r in ['Seoul', 'Gyeonggi'] else 'blue' 
                          for r in region_effects.index])
    axes[1, 1].set_xlabel('Region')
    axes[1, 1].set_ylabel('Consumption Change')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6.6 Summary statistics
    summary_text = f"""
    DID Analysis Results
    ==================
    Relief fund effect: {did_effect:.2f}
    Standard error: {did_se:.2f}
    t-statistic: {did_results.tvalues['treatment']:.2f}
    p-value: {did_results.pvalues['treatment']:.4f}
    
    95% Confidence Interval:
    - Robust SE: [{did_results.conf_int().loc['treatment', 0]:.2f}, 
                  {did_results.conf_int().loc['treatment', 1]:.2f}]
    - Bootstrap: [{ci_lower:.2f}, {ci_upper:.2f}]
    
    Two-way FE: {fe_results.params['treatment']:.2f}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=9, 
                    verticalalignment='center', family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('did_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save data
    panel_df.to_csv('../../data/chapter02_panel_data.csv', index=False)
    print("\nPanel data saved: ../../data/chapter02_panel_data.csv")

if __name__ == "__main__":
    main()