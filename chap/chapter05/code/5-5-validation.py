#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제5장: 모델 검증 시스템
교차검증 기반 모델 성능 평가 및 검증
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_validate,
    TimeSeriesSplit, StratifiedKFold, KFold
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer, mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

class ModelValidationSystem:
    """모델 검증 시스템 클래스"""

    def __init__(self, cv_folds=5, random_state=42):
        """
        모델 검증 시스템 초기화

        Parameters:
        cv_folds (int): 교차검증 폴드 수
        random_state (int): 재현성을 위한 시드
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.validation_results = {}
        self.cv_results = {}
        self.scaler = StandardScaler()

    def generate_validation_data(self, n_samples=1200, n_features=6, scenario='mixed'):
        """
        모델 검증을 위한 시뮬레이션 데이터 생성
        ※ 본 데이터는 교육 목적의 시뮬레이션 데이터입니다

        Parameters:
        n_samples (int): 샘플 수
        n_features (int): 특성 수
        scenario (str): 데이터 시나리오 ('linear', 'nonlinear', 'mixed')

        Returns:
        tuple: (X, y, feature_names) 특성, 타겟, 특성명
        """
        np.random.seed(self.random_state)

        feature_names = [
            '정책투자규모', '경제여건지수', '사회인프라지수',
            '기술혁신지수', '인적자원지수', '환경지속성지수'
        ]

        # 기본 특성 생성
        X = np.random.randn(n_samples, n_features)

        # 시나리오별 타겟 변수 생성
        if scenario == 'linear':
            # 선형 관계
            weights = np.array([2.5, 1.8, 1.2, 0.9, 1.5, 0.7])
            y = X @ weights + 0.3 * np.random.randn(n_samples)

        elif scenario == 'nonlinear':
            # 비선형 관계
            y = (2.0 * X[:, 0] +
                 1.5 * np.power(X[:, 1], 2) +
                 1.0 * np.sin(3 * X[:, 2]) +
                 0.8 * np.exp(0.5 * X[:, 3]) +
                 0.6 * np.log(np.abs(X[:, 4]) + 1) +
                 0.4 * X[:, 5] +
                 0.5 * np.random.randn(n_samples))

        elif scenario == 'mixed':
            # 혼합 관계 (선형 + 비선형 + 상호작용)
            linear_part = 1.5 * X[:, 0] + 1.0 * X[:, 1]
            nonlinear_part = 0.8 * np.power(X[:, 2], 2) + 0.6 * np.sin(2 * X[:, 3])
            interaction_part = 0.4 * X[:, 4] * X[:, 5]
            noise = 0.4 * np.random.randn(n_samples)

            y = linear_part + nonlinear_part + interaction_part + noise

        else:
            raise ValueError(f"지원하지 않는 시나리오: {scenario}")

        # DataFrame 생성
        X_df = pd.DataFrame(X, columns=feature_names)

        print(f"✅ 모델 검증용 데이터 생성 완료 ({scenario} 시나리오)")
        print(f"   - 샘플 수: {n_samples}")
        print(f"   - 특성 수: {n_features}")
        print(f"   - 타겟 범위: [{y.min():.2f}, {y.max():.2f}]")

        return X_df, y, feature_names

    def create_model_suite(self):
        """
        검증할 모델 모음 생성

        Returns:
        dict: 모델 딕셔너리
        """
        # 다양한 복잡도의 모델들 생성
        self.models = {
            # 선형 모델
            'linear_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ]),

            'ridge_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0, random_state=self.random_state))
            ]),

            'lasso_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Lasso(alpha=0.1, random_state=self.random_state))
            ]),

            # 트리 기반 모델
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),

            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),

            # 비선형 모델
            'svr_rbf': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale'))
            ])
        }

        print(f"✅ 모델 모음 생성 완료: {len(self.models)}개 모델")
        for name in self.models.keys():
            print(f"   - {name}")

        return self.models

    def cross_validation_analysis(self, X, y, cv_type='kfold'):
        """
        교차검증 분석 수행

        Parameters:
        X (DataFrame): 입력 특성
        y (array): 타겟 변수
        cv_type (str): 교차검증 타입 ('kfold', 'stratified', 'timeseries')

        Returns:
        dict: 교차검증 결과
        """
        print(f"\n🔄 교차검증 분석 시작 ({cv_type}, {self.cv_folds}-fold)")

        # 교차검증 전략 선택
        if cv_type == 'kfold':
            cv_strategy = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        elif cv_type == 'timeseries':
            cv_strategy = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            raise ValueError(f"지원하지 않는 교차검증 타입: {cv_type}")

        # 평가 지표 정의
        scoring = {
            'mse': make_scorer(mean_squared_error),
            'mae': make_scorer(mean_absolute_error),
            'r2': 'r2',
            'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
        }

        cv_results = {}

        for name, model in self.models.items():
            print(f"   🔄 {name} 교차검증 중...")

            # 교차검증 수행
            cv_scores = cross_validate(
                model, X, y,
                cv=cv_strategy,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

            # 결과 정리
            cv_results[name] = {
                'test_mse': cv_scores['test_mse'],
                'test_mae': cv_scores['test_mae'],
                'test_r2': cv_scores['test_r2'],
                'test_mape': -cv_scores['test_mape'],  # 원래 부호로 복원
                'train_mse': cv_scores['train_mse'],
                'train_mae': cv_scores['train_mae'],
                'train_r2': cv_scores['train_r2'],
                'train_mape': -cv_scores['train_mape'],
                'fit_time': cv_scores['fit_time'],
                'score_time': cv_scores['score_time']
            }

            # 요약 통계 출력
            test_r2_mean = cv_results[name]['test_r2'].mean()
            test_r2_std = cv_results[name]['test_r2'].std()

            print(f"      ✅ 완료 - R²: {test_r2_mean:.4f} (±{test_r2_std:.4f})")

        self.cv_results = cv_results

        print("✅ 모든 모델 교차검증 완료!")

        return cv_results

    def holdout_validation(self, X, y, test_size=0.2):
        """
        홀드아웃 검증 수행

        Parameters:
        X (DataFrame): 입력 특성
        y (array): 타겟 변수
        test_size (float): 테스트 세트 비율

        Returns:
        dict: 홀드아웃 검증 결과
        """
        print(f"\n📊 홀드아웃 검증 시작 (테스트 비율: {test_size})")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        holdout_results = {}

        for name, model in self.models.items():
            print(f"   🔄 {name} 홀드아웃 검증 중...")

            # 모델 학습
            model.fit(X_train, y_train)

            # 예측
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # 평가 지표 계산
            holdout_results[name] = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'test_r2': r2_score(y_test, test_pred),
                'overfit_score': r2_score(y_train, train_pred) - r2_score(y_test, test_pred),
                'predictions': test_pred,
                'actual': y_test
            }

            test_r2 = holdout_results[name]['test_r2']
            overfit = holdout_results[name]['overfit_score']

            print(f"      ✅ 완료 - R²: {test_r2:.4f}, 과적합: {overfit:.4f}")

        self.validation_results = holdout_results

        print("✅ 모든 모델 홀드아웃 검증 완료!")

        return holdout_results

    def analyze_model_stability(self, X, y, n_runs=10):
        """
        모델 안정성 분석 (여러 번 실행)

        Parameters:
        X (DataFrame): 입력 특성
        y (array): 타겟 변수
        n_runs (int): 실행 횟수

        Returns:
        dict: 안정성 분석 결과
        """
        print(f"\n🎯 모델 안정성 분석 시작 ({n_runs}회 실행)")

        stability_results = {}

        for name in self.models.keys():
            stability_results[name] = {
                'test_r2_scores': [],
                'test_mse_scores': [],
                'train_r2_scores': [],
                'train_mse_scores': []
            }

        for run in range(n_runs):
            print(f"   🔄 실행 {run+1}/{n_runs}")

            # 매번 다른 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state + run
            )

            for name, model in self.models.items():
                # 모델 복사 및 학습 (Pipeline과 일반 모델 구분)
                if hasattr(model, 'steps'):  # Pipeline 객체인 경우
                    from sklearn.base import clone
                    model_copy = clone(model)
                else:
                    model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)

                # 예측 및 평가
                train_pred = model_copy.predict(X_train)
                test_pred = model_copy.predict(X_test)

                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)

                stability_results[name]['train_r2_scores'].append(train_r2)
                stability_results[name]['test_r2_scores'].append(test_r2)
                stability_results[name]['train_mse_scores'].append(train_mse)
                stability_results[name]['test_mse_scores'].append(test_mse)

        # 안정성 지표 계산
        for name in self.models.keys():
            test_r2_scores = np.array(stability_results[name]['test_r2_scores'])

            stability_results[name]['mean_test_r2'] = test_r2_scores.mean()
            stability_results[name]['std_test_r2'] = test_r2_scores.std()
            stability_results[name]['cv_test_r2'] = test_r2_scores.std() / test_r2_scores.mean()  # 변동계수

            print(f"   📊 {name}: R² = {test_r2_scores.mean():.4f} ± {test_r2_scores.std():.4f}")

        print("✅ 모델 안정성 분석 완료!")

        return stability_results

    def print_validation_summary(self):
        """검증 결과 요약 출력"""
        if not self.cv_results and not self.validation_results:
            print("⚠️ 검증 결과가 없습니다. 먼저 검증을 수행하세요.")
            return

        print("\n" + "="*80)
        print("📊 모델 검증 결과 종합 요약")
        print("="*80)

        # 교차검증 결과
        if self.cv_results:
            print("\n🔄 교차검증 결과:")
            print("-" * 60)
            for name, results in self.cv_results.items():
                test_r2 = results['test_r2']
                test_mse = results['test_mse']

                print(f"\n🔹 {name.upper()}:")
                print(f"  R² (테스트):  {test_r2.mean():.4f} ± {test_r2.std():.4f}")
                print(f"  MSE (테스트): {test_mse.mean():.4f} ± {test_mse.std():.4f}")
                print(f"  평균 학습시간: {results['fit_time'].mean():.3f}초")

        # 홀드아웃 검증 결과
        if self.validation_results:
            print("\n📊 홀드아웃 검증 결과:")
            print("-" * 60)
            for name, results in self.validation_results.items():
                print(f"\n🔹 {name.upper()}:")
                print(f"  R² (학습):    {results['train_r2']:.4f}")
                print(f"  R² (테스트):  {results['test_r2']:.4f}")
                print(f"  과적합 지표:   {results['overfit_score']:.4f}")

        # 최고 성능 모델 식별
        if self.cv_results:
            best_model_cv = max(self.cv_results.keys(),
                               key=lambda x: self.cv_results[x]['test_r2'].mean())
            print(f"\n🏆 교차검증 최고 성능: {best_model_cv}")

        if self.validation_results:
            best_model_holdout = max(self.validation_results.keys(),
                                   key=lambda x: self.validation_results[x]['test_r2'])
            print(f"🏆 홀드아웃 최고 성능: {best_model_holdout}")

    def plot_validation_results(self, save_path='practice/chapter05/outputs/validation_results.png'):
        """
        검증 결과 시각화

        Parameters:
        save_path (str): 저장 경로
        """
        if not self.cv_results and not self.validation_results:
            print("⚠️ 시각화할 검증 결과가 없습니다.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 교차검증 R² 박스플롯
        if self.cv_results:
            ax1 = axes[0, 0]
            cv_data = []
            cv_labels = []

            for name, results in self.cv_results.items():
                cv_data.append(results['test_r2'])
                cv_labels.append(name)

            bp1 = ax1.boxplot(cv_data, labels=cv_labels, patch_artist=True)
            ax1.set_title('교차검증 R² 분포')
            ax1.set_ylabel('R² 점수')
            ax1.tick_params(axis='x', rotation=45)

            # 박스플롯 색상 설정
            colors = plt.cm.Set3(np.linspace(0, 1, len(cv_data)))
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)

        # 2. 홀드아웃 검증 학습 vs 테스트 성능
        if self.validation_results:
            ax2 = axes[0, 1]
            models = list(self.validation_results.keys())
            train_r2 = [self.validation_results[model]['train_r2'] for model in models]
            test_r2 = [self.validation_results[model]['test_r2'] for model in models]

            x_pos = np.arange(len(models))
            width = 0.35

            ax2.bar(x_pos - width/2, train_r2, width, label='학습 R²', alpha=0.8)
            ax2.bar(x_pos + width/2, test_r2, width, label='테스트 R²', alpha=0.8)

            ax2.set_title('학습 vs 테스트 성능')
            ax2.set_ylabel('R² 점수')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.legend()

        # 3. 과적합 분석
        if self.validation_results:
            ax3 = axes[1, 0]
            models = list(self.validation_results.keys())
            overfit_scores = [self.validation_results[model]['overfit_score'] for model in models]

            bars = ax3.bar(models, overfit_scores, alpha=0.7)
            ax3.set_title('과적합 분석')
            ax3.set_ylabel('과적합 점수 (학습R² - 테스트R²)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # 색상 설정 (과적합 정도에 따라)
            for bar, score in zip(bars, overfit_scores):
                if score > 0.1:
                    bar.set_color('red')
                elif score > 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')

        # 4. 모델 실행 시간 비교 (교차검증 기준)
        if self.cv_results:
            ax4 = axes[1, 1]
            models = list(self.cv_results.keys())
            fit_times = [self.cv_results[model]['fit_time'].mean() for model in models]

            bars = ax4.bar(models, fit_times, alpha=0.7, color='skyblue')
            ax4.set_title('평균 학습 시간')
            ax4.set_ylabel('시간 (초)')
            ax4.tick_params(axis='x', rotation=45)

            # 시간 값 표시
            for bar, time in zip(bars, fit_times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time:.3f}s', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 검증 결과 시각화 저장: {save_path}")

    def generate_validation_report(self, save_path='practice/chapter05/outputs/validation_report.txt'):
        """
        상세 검증 보고서 생성

        Parameters:
        save_path (str): 보고서 저장 경로
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("모델 검증 시스템 종합 보고서\n")
            f.write("="*80 + "\n\n")

            # 교차검증 결과 상세
            if self.cv_results:
                f.write("1. 교차검증 결과 상세\n")
                f.write("-" * 40 + "\n")

                for name, results in self.cv_results.items():
                    f.write(f"\n[{name.upper()}]\n")
                    f.write(f"  테스트 R²:   {results['test_r2'].mean():.4f} ± {results['test_r2'].std():.4f}\n")
                    f.write(f"  테스트 MSE:  {results['test_mse'].mean():.4f} ± {results['test_mse'].std():.4f}\n")
                    f.write(f"  테스트 MAE:  {results['test_mae'].mean():.4f} ± {results['test_mae'].std():.4f}\n")
                    f.write(f"  평균 학습시간: {results['fit_time'].mean():.3f}초\n")

            # 홀드아웃 검증 결과 상세
            if self.validation_results:
                f.write("\n\n2. 홀드아웃 검증 결과 상세\n")
                f.write("-" * 40 + "\n")

                for name, results in self.validation_results.items():
                    f.write(f"\n[{name.upper()}]\n")
                    f.write(f"  학습 R²:     {results['train_r2']:.4f}\n")
                    f.write(f"  테스트 R²:   {results['test_r2']:.4f}\n")
                    f.write(f"  학습 MSE:    {results['train_mse']:.4f}\n")
                    f.write(f"  테스트 MSE:  {results['test_mse']:.4f}\n")
                    f.write(f"  과적합 지표: {results['overfit_score']:.4f}\n")

            # 권장사항
            f.write("\n\n3. 모델 선택 권장사항\n")
            f.write("-" * 40 + "\n")

            if self.cv_results:
                best_cv = max(self.cv_results.keys(),
                             key=lambda x: self.cv_results[x]['test_r2'].mean())
                f.write(f"• 교차검증 기준 최고 성능: {best_cv}\n")

            if self.validation_results:
                best_holdout = max(self.validation_results.keys(),
                                 key=lambda x: self.validation_results[x]['test_r2'])
                f.write(f"• 홀드아웃 기준 최고 성능: {best_holdout}\n")

                # 과적합이 적은 모델 추천
                stable_models = [name for name, results in self.validation_results.items()
                               if results['overfit_score'] < 0.05]
                if stable_models:
                    f.write(f"• 안정성이 우수한 모델: {', '.join(stable_models)}\n")

        print(f"📄 검증 보고서 저장: {save_path}")

def main():
    """메인 실행 함수"""
    print("🚀 모델 검증 시스템 시작")
    print("="*60)

    # 검증 시스템 객체 생성
    validator = ModelValidationSystem(cv_folds=5)

    # 1. 검증용 데이터 생성 (여러 시나리오)
    scenarios = ['linear', 'nonlinear', 'mixed']

    for scenario in scenarios:
        print(f"\n📋 {scenario.upper()} 시나리오 검증")
        print("="*50)

        # 데이터 생성
        X, y, feature_names = validator.generate_validation_data(
            n_samples=1200, n_features=6, scenario=scenario
        )

        # 2. 모델 모음 생성
        print("\n🤖 모델 모음 생성")
        validator.create_model_suite()

        # 3. 교차검증 분석
        print("\n🔄 교차검증 분석")
        cv_results = validator.cross_validation_analysis(X, y, cv_type='kfold')

        # 4. 홀드아웃 검증
        print("\n📊 홀드아웃 검증")
        holdout_results = validator.holdout_validation(X, y, test_size=0.2)

        # 5. 안정성 분석
        print("\n🎯 모델 안정성 분석")
        stability_results = validator.analyze_model_stability(X, y, n_runs=5)

        # 6. 결과 요약 출력
        validator.print_validation_summary()

        # 7. 시각화 (마지막 시나리오만)
        if scenario == scenarios[-1]:
            print("\n📈 검증 결과 시각화")
            validator.plot_validation_results()

            # 8. 보고서 생성
            print("\n📄 검증 보고서 생성")
            validator.generate_validation_report()

    # 9. 최종 요약
    print("\n" + "="*60)
    print("🎯 모델 검증 시스템 완료!")
    print("="*60)

    print(f"📊 검증 완료:")
    print(f"   - 시나리오: {len(scenarios)}개")
    print(f"   - 모델: {len(validator.models)}개")
    print(f"   - 교차검증: {validator.cv_folds}-fold")

    print(f"\n📁 생성된 파일:")
    print("   - practice/chapter05/outputs/validation_results.png")
    print("   - practice/chapter05/outputs/validation_report.txt")

    print("\n✅ 모든 모델 검증이 완료되었습니다!")

if __name__ == "__main__":
    main()