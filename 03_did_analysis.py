
# 간단한 DID 분석: 데이터 불러오기 → DID 회귀분석
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = pd.read_csv('chapter02_panel_data.csv')

# DID 회귀분석
model = smf.ols('consumption ~ treated + post + treatment', data=df)
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['unit_id']})

print(result.summary().tables[1])
print(f"\n긴급재난지원금 효과(DID): {result.params['treatment']:.2f} (SE: {result.bse['treatment']:.2f})")

# 1. 처리집단/비처리집단의 월별 평균 소비 시계열 그래프
treated_mean = df[df['treated']==1].groupby('time')['consumption'].mean()
control_mean = df[df['treated']==0].groupby('time')['consumption'].mean()
plt.figure(figsize=(8,5))
plt.plot(treated_mean.index, treated_mean.values, 'o-', label='Treated', linewidth=2)
plt.plot(control_mean.index, control_mean.values, 's-', label='Control', linewidth=2)
plt.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Relief Fund Payment')
plt.xlabel('Month')
plt.ylabel('Average Consumption')
plt.title('Time Trend of Consumption by Group')
plt.legend()
plt.tight_layout()
plt.savefig('did_trend.png', dpi=200)
plt.show()

# 2. DID 전후(before/after) 평균 소비 변화 도표
pre_post = df.groupby(['treated', 'post'])['consumption'].mean().unstack()
plt.figure(figsize=(6,5))
plt.plot([0,1], pre_post.loc[1], 'o-', label='Treated', linewidth=2)
plt.plot([0,1], pre_post.loc[0], 's-', label='Control', linewidth=2)
plt.xticks([0,1], ['Before', 'After'])
plt.ylabel('Average Consumption')
plt.title('DID: Before/After by Group')
plt.legend()
plt.tight_layout()
plt.savefig('did_before_after.png', dpi=200)
plt.show()