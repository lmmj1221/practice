"""
최적 모델 선택을 위한 체계적 방법론
Author: AI Policy Analyst
Date: 2024

목적: 정책 분석을 위한 최적 모델을 과학적으로 선택하는 프레임워크
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


class OptimalModelSelector:
    """
    정책 분석을 위한 최적 모델 선택 프레임워크
    """

    def __init__(self, objective='balanced'):
        """
        Parameters:
        -----------
        objective : str
            'explanation' - 설명력 중심
            'prediction' - 예측력 중심
            'balanced' - 균형 추구
            'speed' - 실시간 처리 중심
        """
        self.objective = objective
        self.results = {}

        # 목적별 가중치 설정
        self.weights = self._set_weights(objective)

        # 후보 모델 정의
        self.models = self._initialize_models()

    def _set_weights(self, objective):
        """목적별 평가 기준 가중치 설정"""
        weight_profiles = {
            'explanation': {
                'accuracy': 0.2,
                'explainability': 0.4,
                'simplicity': 0.2,
                'speed': 0.1,
                'robustness': 0.1
            },
            'prediction': {
                'accuracy': 0.5,
                'explainability': 0.1,
                'simplicity': 0.05,
                'speed': 0.15,
                'robustness': 0.2
            },
            'balanced': {
                'accuracy': 0.3,
                'explainability': 0.3,
                'simplicity': 0.15,
                'speed': 0.1,
                'robustness': 0.15
            },
            'speed': {
                'accuracy': 0.2,
                'explainability': 0.1,
                'simplicity': 0.2,
                'speed': 0.4,
                'robustness': 0.1
            }
        }
        return weight_profiles[objective]

    def _initialize_models(self):
        """후보 모델 초기화"""
        return {
            'Linear Regression': {
                'model': LinearRegression(),
                'explainability': 1.0,  # 완전 설명 가능
                'simplicity': 1.0,
                'category': 'linear'
            },
            'Ridge Regression': {
                'model': Ridge(alpha=1.0),
                'explainability': 0.95,
                'simplicity': 0.95,
                'category': 'linear'
            },
            'Lasso Regression': {
                'model': Lasso(alpha=0.1),
                'explainability': 0.9,
                'simplicity': 0.9,
                'category': 'linear'
            },
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'explainability': 0.6,  # 특성 중요도만 제공
                'simplicity': 0.4,
                'category': 'ensemble'
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'explainability': 0.5,
                'simplicity': 0.3,
                'category': 'ensemble'
            },
            'Neural Network': {
                'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
                'explainability': 0.2,  # 블랙박스
                'simplicity': 0.1,
                'category': 'deep'
            }
        }

    def evaluate_models(self, X, y, cv=5):
        """
        모든 모델을 평가하고 점수 계산
        """
        print(f"\n{'='*60}")
        print(f"목적: {self.objective.upper()} 최적화")
        print(f"{'='*60}\n")

        # 데이터 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for name, config in self.models.items():
            print(f"평가 중: {name}")
            model = config['model']

            # 1. 정확도 평가 (교차검증)
            scores = cross_val_score(model, X_scaled, y, cv=cv,
                                    scoring='r2')
            accuracy = scores.mean()
            robustness = 1 - scores.std()  # 안정성

            # 2. 속도 평가
            import time
            start = time.time()
            model.fit(X_scaled, y)
            train_time = time.time() - start

            start = time.time()
            _ = model.predict(X_scaled[:100])
            pred_time = time.time() - start

            # 속도 점수 (역수 정규화)
            speed_score = 1 / (1 + train_time + pred_time * 10)

            # 3. 종합 점수 계산
            total_score = (
                self.weights['accuracy'] * accuracy +
                self.weights['explainability'] * config['explainability'] +
                self.weights['simplicity'] * config['simplicity'] +
                self.weights['speed'] * speed_score +
                self.weights['robustness'] * robustness
            )

            # 결과 저장
            self.results[name] = {
                'accuracy': accuracy,
                'explainability': config['explainability'],
                'simplicity': config['simplicity'],
                'speed': speed_score,
                'robustness': robustness,
                'total_score': total_score,
                'train_time': train_time,
                'category': config['category']
            }

            print(f"  - R² Score: {accuracy:.4f}")
            print(f"  - 종합 점수: {total_score:.4f}\n")

    def find_pareto_optimal(self):
        """
        파레토 최적 모델들 찾기
        (정확도와 설명력의 trade-off)
        """
        pareto_models = []

        for name1, metrics1 in self.results.items():
            is_dominated = False

            for name2, metrics2 in self.results.items():
                if name1 == name2:
                    continue

                # name2가 name1을 지배하는지 확인
                if (metrics2['accuracy'] > metrics1['accuracy'] and
                    metrics2['explainability'] >= metrics1['explainability']) or \
                   (metrics2['accuracy'] >= metrics1['accuracy'] and
                    metrics2['explainability'] > metrics1['explainability']):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_models.append(name1)

        return pareto_models

    def recommend_model(self):
        """
        최종 모델 추천
        """
        # 1. 종합 점수 최고 모델
        best_overall = max(self.results.items(),
                          key=lambda x: x[1]['total_score'])

        # 2. 파레토 최적 모델들
        pareto_models = self.find_pareto_optimal()

        # 3. 카테고리별 최고 모델
        category_best = {}
        for name, metrics in self.results.items():
            cat = metrics['category']
            if cat not in category_best or \
               metrics['total_score'] > category_best[cat][1]:
                category_best[cat] = (name, metrics['total_score'])

        print("\n" + "="*60)
        print("📊 모델 선택 결과")
        print("="*60)

        print(f"\n✅ 종합 최적 모델: {best_overall[0]}")
        print(f"   종합 점수: {best_overall[1]['total_score']:.4f}")
        print(f"   정확도: {best_overall[1]['accuracy']:.4f}")
        print(f"   설명력: {best_overall[1]['explainability']:.2f}")

        print(f"\n🎯 파레토 최적 모델들 (정확도-설명력):")
        for model in pareto_models:
            print(f"   - {model}: 정확도={self.results[model]['accuracy']:.3f}, "
                  f"설명력={self.results[model]['explainability']:.2f}")

        print(f"\n📈 카테고리별 최고 모델:")
        for cat, (model, score) in category_best.items():
            print(f"   - {cat}: {model} (점수: {score:.4f})")

        # 최종 추천
        print(f"\n💡 최종 추천:")
        if self.objective == 'explanation':
            # 설명력 우선시 파레토 최적 중 설명력 높은 모델
            pareto_explain = [(m, self.results[m]['explainability'])
                             for m in pareto_models]
            recommended = max(pareto_explain, key=lambda x: x[1])[0]
        else:
            recommended = best_overall[0]

        print(f"   목적 '{self.objective}'에 가장 적합한 모델: {recommended}")

        return recommended, self.results

    def visualize_results(self):
        """
        모델 선택 결과 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 종합 점수 비교
        models = list(self.results.keys())
        scores = [self.results[m]['total_score'] for m in models]

        axes[0, 0].barh(models, scores, color='steelblue')
        axes[0, 0].set_xlabel('종합 점수')
        axes[0, 0].set_title(f'모델별 종합 점수 ({self.objective} 최적화)')
        axes[0, 0].axvline(x=max(scores), color='red', linestyle='--', alpha=0.5)

        # 2. 정확도 vs 설명력 (파레토 프론트)
        accuracy = [self.results[m]['accuracy'] for m in models]
        explainability = [self.results[m]['explainability'] for m in models]

        pareto_models = self.find_pareto_optimal()

        axes[0, 1].scatter(accuracy, explainability, s=100, alpha=0.6)
        for i, model in enumerate(models):
            color = 'red' if model in pareto_models else 'black'
            weight = 'bold' if model in pareto_models else 'normal'
            axes[0, 1].annotate(model, (accuracy[i], explainability[i]),
                               fontsize=8, color=color, weight=weight)

        axes[0, 1].set_xlabel('정확도 (R² Score)')
        axes[0, 1].set_ylabel('설명력')
        axes[0, 1].set_title('정확도 vs 설명력 (빨간색: 파레토 최적)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 다차원 평가 레이더 차트
        categories = ['정확도', '설명력', '단순성', '속도', '강건성']

        # 상위 3개 모델만 표시
        top_models = sorted(self.results.items(),
                          key=lambda x: x[1]['total_score'],
                          reverse=True)[:3]

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(223, projection='polar')

        for name, metrics in top_models:
            values = [
                metrics['accuracy'],
                metrics['explainability'],
                metrics['simplicity'],
                metrics['speed'],
                metrics['robustness']
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Top 3 모델 다차원 비교')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        # 4. 가중치 영향 분석
        weight_keys = list(self.weights.keys())
        weight_values = list(self.weights.values())

        axes[1, 1].pie(weight_values, labels=weight_keys, autopct='%1.1f%%',
                      startangle=90, colors=plt.cm.Set3.colors)
        axes[1, 1].set_title(f'평가 기준 가중치 ({self.objective} 모드)')

        plt.tight_layout()

        # 저장
        output_dir = 'c:/practice/chap/chapter05/outputs'
        import os
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/model_selection_{self.objective}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

        return fig


def demonstrate_model_selection():
    """
    다양한 데이터 패턴에서 최적 모델 선택 시연
    """
    np.random.seed(42)

    # 시나리오별 데이터 생성
    scenarios = {
        '선형 패턴': lambda X: 2*X[:, 0] + 3*X[:, 1] - X[:, 2],
        '비선형 패턴': lambda X: np.sin(X[:, 0]) + X[:, 1]**2 + np.log(np.abs(X[:, 2]) + 1),
        '복잡한 상호작용': lambda X: X[:, 0]*X[:, 1] + np.exp(-X[:, 2]) + X[:, 3]**3
    }

    n_samples = 1000
    n_features = 10

    results_summary = {}

    for scenario_name, target_func in scenarios.items():
        print(f"\n{'='*60}")
        print(f"시나리오: {scenario_name}")
        print(f"{'='*60}")

        # 데이터 생성
        X = np.random.randn(n_samples, n_features)
        y = target_func(X) + np.random.randn(n_samples) * 0.1

        # 각 목적별로 최적 모델 선택
        objectives = ['explanation', 'prediction', 'balanced']
        scenario_results = {}

        for obj in objectives:
            selector = OptimalModelSelector(objective=obj)
            selector.evaluate_models(X, y)
            recommended, all_results = selector.recommend_model()

            scenario_results[obj] = {
                'recommended': recommended,
                'score': all_results[recommended]['total_score'],
                'accuracy': all_results[recommended]['accuracy']
            }

            # 첫 번째 목적에 대해서만 시각화
            if obj == 'balanced':
                selector.visualize_results()

        results_summary[scenario_name] = scenario_results

    # 최종 요약
    print("\n" + "="*60)
    print("📊 전체 시나리오 요약")
    print("="*60)

    summary_df = pd.DataFrame()
    for scenario, objectives in results_summary.items():
        for obj, metrics in objectives.items():
            row = {
                '시나리오': scenario,
                '목적': obj,
                '추천 모델': metrics['recommended'],
                '정확도': f"{metrics['accuracy']:.4f}",
                '종합 점수': f"{metrics['score']:.4f}"
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([row])],
                                  ignore_index=True)

    print(summary_df.to_string(index=False))

    # 결과 저장
    output_dir = 'c:/practice/chap/chapter05/outputs'
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(f'{output_dir}/model_selection_summary.csv',
                     index=False, encoding='utf-8-sig')

    print(f"\n✅ 결과가 {output_dir}에 저장되었습니다.")

    return results_summary


if __name__ == "__main__":
    print("🚀 최적 모델 선택 프레임워크 시작")
    results = demonstrate_model_selection()
    print("\n✅ 모든 분석이 완료되었습니다!")