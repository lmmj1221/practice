#!/usr/bin/env python3
"""
하이브리드 모델링: 머신러닝 + 딥러닝 결합 구현
챕터 5 - feature_fusion.py
생성일: 2025-09-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """데이터 로드 및 전처리"""
    # 샘플 데이터 생성 (실제로는 CSV 로드)
    np.random.seed(42)
    n_samples = 1000

    data = {
        'gdp_growth': np.random.normal(2.5, 1.5, n_samples),
        'inflation': np.random.normal(2.0, 0.5, n_samples),
        'unemployment': np.random.normal(3.5, 1.0, n_samples),
        'policy_effect': np.random.normal(0.5, 0.2, n_samples)
    }

    return pd.DataFrame(data)


def train_model(X, y):
    """모델 학습"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    print(f"모델 성능 (R2 Score): {score:.4f}")
    return model


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print(f"하이브리드 모델링: 머신러닝 + 딥러닝 결합 실습")
    print("=" * 50)

    # 데이터 로드
    df = load_data()
    print(f"\n데이터 shape: {df.shape}")

    # 특징과 타겟 분리
    X = df[['gdp_growth', 'inflation', 'unemployment']]
    y = df['policy_effect']

    # 모델 학습
    model = train_model(X, y)

    # 예측
    sample_input = [[2.5, 2.0, 3.5]]
    prediction = model.predict(sample_input)
    print(f"\n예측 결과: {prediction[0]:.4f}")

    print("\n실습 완료!")


if __name__ == "__main__":
    main()
