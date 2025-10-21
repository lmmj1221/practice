"""
최적 모델 선택 - 간단 시연 버전
빠른 실행을 위해 최적화된 버전
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def quick_model_selection_demo():
    """빠른 모델 선택 시연"""

    print("="*60)
    print("🚀 최적 모델 선택 프레임워크 - 간단 시연")
    print("="*60)

    # 1. 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 500
    n_features = 5

    # 선형 패턴 데이터
    X_linear = np.random.randn(n_samples, n_features)
    y_linear = 2*X_linear[:, 0] + 3*X_linear[:, 1] + np.random.randn(n_samples) * 0.1

    # 비선형 패턴 데이터
    X_nonlinear = np.random.randn(n_samples, n_features)
    y_nonlinear = np.sin(X_nonlinear[:, 0]) + X_nonlinear[:, 1]**2 + np.random.randn(n_samples) * 0.1

    # 2. 모델 정의
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'explainability': 1.0,
            'simplicity': 1.0
        },
        'Ridge Regression': {
            'model': Ridge(alpha=1.0),
            'explainability': 0.95,
            'simplicity': 0.95
        },
        'Random Forest': {
            'model': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'explainability': 0.6,
            'simplicity': 0.4
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'explainability': 0.5,
            'simplicity': 0.3
        }
    }

    # 3. 가중치 프로파일
    weight_profiles = {
        '설명력 우선': {'accuracy': 0.2, 'explainability': 0.4, 'simplicity': 0.2, 'speed': 0.1, 'robustness': 0.1},
        '예측력 우선': {'accuracy': 0.5, 'explainability': 0.1, 'simplicity': 0.05, 'speed': 0.15, 'robustness': 0.2},
        '균형 추구': {'accuracy': 0.3, 'explainability': 0.3, 'simplicity': 0.15, 'speed': 0.1, 'robustness': 0.15}
    }

    results = []

    # 4. 모델 평가
    print("\n📊 시나리오별 최적 모델 선택")
    print("-"*60)

    scenarios = [
        ('선형 패턴', X_linear, y_linear),
        ('비선형 패턴', X_nonlinear, y_nonlinear)
    ]

    for scenario_name, X, y in scenarios:
        print(f"\n🔍 {scenario_name} 데이터")
        print("-"*40)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scenario_results = {}

        for profile_name, weights in weight_profiles.items():
            best_score = -1
            best_model = None

            for model_name, config in models.items():
                model = config['model']

                # 정확도 평가 (3-fold로 축소)
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
                accuracy = scores.mean()
                robustness = 1 - scores.std()

                # 속도 점수 (단순화)
                import time
                start = time.time()
                model.fit(X_scaled, y)
                train_time = time.time() - start
                speed_score = 1 / (1 + train_time * 10)

                # 종합 점수
                total_score = (
                    weights['accuracy'] * max(0, accuracy) +
                    weights['explainability'] * config['explainability'] +
                    weights['simplicity'] * config['simplicity'] +
                    weights['speed'] * speed_score +
                    weights['robustness'] * robustness
                )

                if total_score > best_score:
                    best_score = total_score
                    best_model = model_name
                    best_accuracy = accuracy

            scenario_results[profile_name] = {
                'model': best_model,
                'score': best_score,
                'accuracy': best_accuracy
            }

            print(f"  {profile_name:12} → {best_model:20} (R²={best_accuracy:.3f}, 점수={best_score:.3f})")

        results.append({
            'scenario': scenario_name,
            'results': scenario_results
        })

    # 5. 파레토 최적 분석
    print("\n" + "="*60)
    print("🎯 파레토 최적 모델 분석")
    print("-"*60)

    # 마지막 시나리오(비선형)에 대한 파레토 분석
    X, y = X_nonlinear, y_nonlinear
    X_scaled = scaler.fit_transform(X)

    model_performance = []
    for model_name, config in models.items():
        model = config['model']
        scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
        accuracy = scores.mean()

        model_performance.append({
            'name': model_name,
            'accuracy': max(0, accuracy),
            'explainability': config['explainability']
        })

    # 파레토 최적 찾기
    pareto_optimal = []
    for i, model1 in enumerate(model_performance):
        is_dominated = False
        for j, model2 in enumerate(model_performance):
            if i == j:
                continue
            if (model2['accuracy'] > model1['accuracy'] and model2['explainability'] >= model1['explainability']) or \
               (model2['accuracy'] >= model1['accuracy'] and model2['explainability'] > model1['explainability']):
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(model1['name'])
            print(f"✅ {model1['name']:20} - 정확도: {model1['accuracy']:.3f}, 설명력: {model1['explainability']:.2f}")

    # 6. 시각화
    create_visualization(model_performance, results)

    # 7. 요약
    print("\n" + "="*60)
    print("💡 핵심 통찰")
    print("-"*60)
    print("1. 선형 데이터 → Linear Regression이 모든 목적에 최적")
    print("2. 비선형 데이터 → 목적에 따라 다른 모델 선택")
    print("3. 파레토 최적 → 정확도-설명력 trade-off 고려")
    print("="*60)

    return results

def create_visualization(model_performance, results):
    """결과 시각화"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 정확도 vs 설명력 산점도
    accuracy = [m['accuracy'] for m in model_performance]
    explainability = [m['explainability'] for m in model_performance]
    names = [m['name'] for m in model_performance]

    axes[0].scatter(accuracy, explainability, s=200, alpha=0.6, c=range(len(names)), cmap='viridis')

    for i, name in enumerate(names):
        axes[0].annotate(name, (accuracy[i], explainability[i]),
                        fontsize=9, ha='center', va='bottom')

    axes[0].set_xlabel('정확도 (R² Score)', fontsize=11)
    axes[0].set_ylabel('설명력', fontsize=11)
    axes[0].set_title('모델별 정확도-설명력 Trade-off', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].set_ylim(-0.1, 1.1)

    # 2. 목적별 최적 모델 막대 그래프
    profiles = ['설명력 우선', '예측력 우선', '균형 추구']
    linear_models = []
    nonlinear_models = []

    for result in results:
        if result['scenario'] == '선형 패턴':
            linear_models = [result['results'][p]['model'] for p in profiles]
        else:
            nonlinear_models = [result['results'][p]['model'] for p in profiles]

    x = np.arange(len(profiles))
    width = 0.35

    # 각 모델을 숫자로 매핑하여 막대 높이로 사용
    model_to_num = {'Linear Regression': 1, 'Ridge Regression': 2,
                    'Random Forest': 3, 'Gradient Boosting': 4}

    linear_nums = [model_to_num.get(m, 0) for m in linear_models]
    nonlinear_nums = [model_to_num.get(m, 0) for m in nonlinear_models]

    bars1 = axes[1].bar(x - width/2, linear_nums, width, label='선형 패턴', alpha=0.8)
    bars2 = axes[1].bar(x + width/2, nonlinear_nums, width, label='비선형 패턴', alpha=0.8)

    axes[1].set_xlabel('목적', fontsize=11)
    axes[1].set_ylabel('모델 유형', fontsize=11)
    axes[1].set_title('목적별 최적 모델 선택', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(profiles)
    axes[1].set_yticks([1, 2, 3, 4])
    axes[1].set_yticklabels(['Linear', 'Ridge', 'Random\nForest', 'Gradient\nBoosting'], fontsize=9)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 저장
    output_dir = 'c:/practice/chap/chapter05/outputs'
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/model_selection_simple.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n📊 시각화가 {output_dir}/model_selection_simple.png에 저장되었습니다.")

if __name__ == "__main__":
    results = quick_model_selection_demo()
    print("\n✅ 모델 선택 분석이 완료되었습니다!")