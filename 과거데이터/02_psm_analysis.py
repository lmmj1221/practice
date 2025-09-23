
# 간단한 PSM 분석: 데이터 불러오기 → 성향점수 추정 → 최근접 매칭 → ATT 계산
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

print("[1] 데이터 불러오기...")
df = pd.read_csv('chapter02_data.csv')
print(f"샘플 수: {len(df)}")

print("[2] 성향점수(Propensity Score) 추정...")
X = df[['ability', 'education']]
y = df['treatment']
ps_model = LogisticRegression(max_iter=1000, random_state=2025)
ps_model.fit(X, y)
df['ps'] = ps_model.predict_proba(X)[:, 1]
print(f"성향점수 범위: {df['ps'].min():.3f} ~ {df['ps'].max():.3f}")

# 매칭 전 성향점수 분포 시각화
plt.figure(figsize=(7,4))
plt.hist(df[df['treatment']==1]['ps'], bins=20, alpha=0.5, label='Treated', density=True)
plt.hist(df[df['treatment']==0]['ps'], bins=20, alpha=0.5, label='Control', density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution (Before Matching)')
plt.legend()
plt.tight_layout()
plt.savefig('psm_ps_before.png', dpi=200)
plt.show()

print("[3] 1:1 최근접 이웃 매칭(caliper=0.1)...")
treated = df[df['treatment'] == 1].copy()
control = df[df['treatment'] == 0].copy()
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['ps']].values)
matches = []
for idx in treated.index:
    ps_treated = df.loc[idx, 'ps']
    distances, indices = nn.kneighbors([[ps_treated]])
    if distances[0][0] <= 0.1:
        control_idx = control.iloc[indices[0][0]].name
        matches.append((idx, control_idx))

print(f"총 매칭 쌍 수: {len(matches)}")
if len(matches) == 0:
    print("매칭된 쌍이 없습니다. caliper를 조정하세요.")
else:
    treated_indices = [m[0] for m in matches]
    control_indices = [m[1] for m in matches]
    att = df.loc[treated_indices, 'income'].mean() - df.loc[control_indices, 'income'].mean()
    print(f"ATT(처리집단 평균효과): {att:.2f}")

    # 매칭 후 성향점수 분포 시각화
    plt.figure(figsize=(7,4))
    plt.hist(df.loc[treated_indices, 'ps'], bins=20, alpha=0.5, label='Treated (Matched)', density=True)
    plt.hist(df.loc[control_indices, 'ps'], bins=20, alpha=0.5, label='Control (Matched)', density=True)
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Distribution (After Matching)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psm_ps_after.png', dpi=200)
    plt.show()

    # 매칭 전후 변수(ability, education) 평균 비교
    variables = ['ability', 'education']
    before_treated = df[df['treatment']==1][variables].mean()
    before_control = df[df['treatment']==0][variables].mean()
    after_treated = df.loc[treated_indices, variables].mean()
    after_control = df.loc[control_indices, variables].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for i, var in enumerate(variables):
        axes[i].bar(['Treated (Before)', 'Control (Before)'], [before_treated[var], before_control[var]], alpha=0.5, label='Before')
        axes[i].bar(['Treated (After)', 'Control (After)'], [after_treated[var], after_control[var]], alpha=0.7, label='After')
        axes[i].set_title(f'{var} Mean')
        axes[i].set_ylabel(var)
    plt.suptitle('Variable Means Before/After Matching')
    plt.tight_layout()
    plt.savefig('psm_var_means.png', dpi=200)
    plt.show()

    # ATT(처리집단 평균효과) 결과 분포 시각화
    pair_effects = [df.loc[t_idx, 'income'] - df.loc[c_idx, 'income'] for t_idx, c_idx in zip(treated_indices, control_indices)]
    plt.figure(figsize=(7,4))
    plt.hist(pair_effects, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=att, color='red', linestyle='--', linewidth=2, label=f'ATT: {att:.2f}')
    plt.xlabel('Income Difference (Treated - Control)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Matched Pair Effects')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psm_att_effect.png', dpi=200)
    plt.show()