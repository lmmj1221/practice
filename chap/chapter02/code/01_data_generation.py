import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
data_dir = current_dir.parent / 'data'
data_dir.mkdir(exist_ok=True)

# 시뮬레이션 데이터 생성을 위한 파라미터 설정
np.random.seed(42)
n_students = 2000  # 전체 학생 수
pro_scholarship_students = 500 # 장학금 수혜 학생 수

# 학생 특성(공변량) 생성
# 학업 능력(ability), 가정 소득(income), 학습 의지(motivation)
ability = np.random.normal(70, 10, n_students)
income = np.random.randint(1, 11, n_students) * 1000 # 1000만원 단위
motivation = np.random.uniform(0, 1, n_students)

# 장학금 수혜 여부(처리 변수 T) 생성
# 성적이 높고 소득이 낮을수록 장학금을 받을 확률이 높다고 가정
propensity = 1 / (1 + np.exp(-(0.1 * ability - 0.3 * (income / 1000) + 2 * motivation - 5)))
scholarship = np.zeros(n_students)
# 장학금 수혜 학생 무작위로 선정
scholarship_indices = np.random.choice(n_students, pro_scholarship_students, p=propensity/propensity.sum(), replace=False)
scholarship[scholarship_indices] = 1


# 결과 변수(Y) 생성: 대학 진학 시험 점수
# 기본 점수 + 학업능력 영향 + 장학금 효과
# 장학금 효과는 학생 특성에 따라 다르게 적용 (이질적 효과)
# 저소득 & 성실한 학생에게 효과가 더 크게 나타남
true_effect = 10 + (10 - income / 1000) * motivation 
score = 50 + 0.5 * ability + scholarship * true_effect + np.random.normal(0, 5, n_students)

# 데이터프레임 생성
df = pd.DataFrame({
    'ability': ability,
    'income': income,
    'motivation': motivation,
    'scholarship': scholarship,
    'score': score
})

# CSV 파일로 저장
df.to_csv(data_dir / 'scholarship_data.csv', index=False)

print(f"시뮬레이션 데이터 생성 완료. 경로: {data_dir / 'scholarship_data.csv'}")
print(df.head())
