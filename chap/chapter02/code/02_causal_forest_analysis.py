import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from econml.dml import CausalForestDML

# Setup paths
current_dir = Path(__file__).parent
data_dir = current_dir.parent / 'data'

# 1. 데이터 불러오기
df = pd.read_csv(data_dir / 'scholarship_data.csv')

# 2. 변수 설정
# 결과 변수 (Y)
Y = df['score']
# 처리 변수 (T)
T = df['scholarship']
# 공변량 (X)
X = df[['ability', 'income', 'motivation']]
# 이질적 효과를 확인할 변수 (W)
W = df[['income', 'motivation']]

# 3. Causal Forest 모델 학습
# 모델 정의: 이질적 효과를 추정하기 위해 CausalForestDML 사용
# n_estimators: 만들 나무의 개수
# max_depth: 나무의 최대 깊이
# min_samples_leaf: 리프 노드가 되기 위한 최소한의 샘플 수
est = CausalForestDML(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)

# 모델 학습
est.fit(Y, T, X=X, W=W)

# 4. 개인별 조건부 평균 처리 효과(CATE) 추정
# 각 학생에 대한 장학금의 예상 효과(점수 향상)
cate_estimates = est.effect(X)
df['cate_estimate'] = cate_estimates

print("Causal Forest 모델 학습 및 CATE 추정 완료.")
print("상위 5개 학생의 추정된 장학금 효과 (CATE):")
print(df[['ability', 'income', 'motivation', 'cate_estimate']].head())

# 5. 효과 시각화
# 소득 수준에 따른 장학금 효과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(df['income'], df['cate_estimate'], alpha=0.5)
plt.title('Income vs. Estimated Scholarship Effect (CATE)')
plt.xlabel('Income (in 10,000 KRW)')
plt.ylabel('Estimated Effect on Score (CATE)')
plt.grid(True)
plt.show()

# 학습 의지에 따른 장학금 효과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(df['motivation'], df['cate_estimate'], alpha=0.5)
plt.title('Motivation vs. Estimated Scholarship Effect (CATE)')
plt.xlabel('Motivation')
plt.ylabel('Estimated Effect on Score (CATE)')
plt.grid(True)
plt.show()

# 6. 이질적 효과 해석
# 소득이 낮을수록, 학습 의지가 높을수록 장학금의 긍정적 효과가 크다는 것을 확인
# 특정 소득 및 학습 의지 그룹에 대한 평균 효과 확인
low_income_high_motivation_effect = df[(df['income'] <= 3000) & (df['motivation'] >= 0.7)]['cate_estimate'].mean()
high_income_low_motivation_effect = df[(df['income'] > 7000) & (df['motivation'] < 0.3)]['cate_estimate'].mean()

print(f"\n저소득 & 높은 학습의지 그룹의 평균 장학금 효과: {low_income_high_motivation_effect:.2f}점 상승")
print(f"고소득 & 낮은 학습의지 그룹의 평균 장학금 효과: {high_income_low_motivation_effect:.2f}점 상승")
