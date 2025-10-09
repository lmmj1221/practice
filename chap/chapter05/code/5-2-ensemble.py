#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제5장: 앙상블 모델 구현
XGBoost와 Random Forest를 결합한 Voting Regressor 구현
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

class EnsembleModelImplementation:
    """앙상블 모델 구현 클래스"""

    def __init__(self, random_state=42):
        """
        앙상블 모델 초기화

        Parameters:
        random_state (int): 재현성을 위한 시드
        """
        self.random_state = random_state
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.best_params = {}

    def generate_policy_data(self, n_samples=1500, n_features=8):
        """
        정책 예측을 위한 시뮬레이션 데이터 생성
        ※ 본 데이터는 교육 목적의 시뮬레이션 데이터입니다

        Parameters:
        n_samples (int): 샘플 수
        n_features (int): 특성 수

        Returns:
        tuple: (X, y, feature_names) 특성, 타겟, 특성명
        """
        np.random.seed(self.random_state)

        # 정책 관련 특성 생성
        feature_names = [
            'GDP Growth', 'Unemployment', 'Inflation', 'Gov Spending',
            'Population Density', 'Education Index', 'Infrastructure', 'Tech Innovation'
        ]

        # 다양한 분포에서 특성 생성
        X = np.zeros((n_samples, n_features))
        X[:, 0] = np.random.normal(2.5, 1.0, n_samples)    # 경제성장률
        X[:, 1] = np.random.exponential(3.5, n_samples)    # 실업률
        X[:, 2] = np.random.normal(2.0, 0.8, n_samples)    # 인플레이션율
        X[:, 3] = np.random.uniform(15, 35, n_samples)     # 정부지출비율
        X[:, 4] = np.random.gamma(2, 2, n_samples)         # 인구밀도
        X[:, 5] = np.random.beta(2, 5, n_samples) * 100    # 교육지수
        X[:, 6] = np.random.lognormal(3, 0.5, n_samples)   # 인프라지수
        X[:, 7] = np.random.weibull(2, n_samples) * 50     # 기술혁신지수

        # 복잡한 비선형 관계로 타겟 변수 생성
        y = (0.8 * X[:, 0] +                               # 경제성장률 직접 효과
             -0.5 * X[:, 1] +                              # 실업률 역효과
             -0.3 * X[:, 2] ** 2 +                         # 인플레이션 제곱 효과
             0.02 * X[:, 3] +                              # 정부지출 효과
             0.1 * np.log(X[:, 4] + 1) +                   # 인구밀도 로그 효과
             0.05 * X[:, 5] +                              # 교육 효과
             0.02 * X[:, 6] +                              # 인프라 효과
             0.03 * X[:, 7] +                              # 기술혁신 효과
             0.1 * X[:, 0] * X[:, 5] +                     # 경제성장-교육 상호작용
             -0.05 * X[:, 1] * X[:, 2] +                   # 실업-인플레이션 상호작용
             0.5 * np.random.randn(n_samples))             # 노이즈

        # DataFrame으로 변환
        X_df = pd.DataFrame(X, columns=feature_names)

        print(f"✅ 정책 시뮬레이션 데이터 생성 완료")
        print(f"   - 샘플 수: {n_samples}")
        print(f"   - 특성 수: {n_features}")
        print(f"   - 타겟 범위: [{y.min():.2f}, {y.max():.2f}]")

        return X_df, y, feature_names

    def create_base_models(self):
        """
        기본 모델들 생성

        Returns:
        dict: 기본 모델 딕셔너리
        """
        # XGBoost 모델
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Random Forest 모델
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Gradient Boosting 모델
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )

        self.models = {
            'xgboost': xgb_model,
            'random_forest': rf_model,
            'gradient_boosting': gb_model
        }

        print("✅ 기본 모델 생성 완료")
        for name in self.models.keys():
            print(f"   - {name}")

        return self.models

    def optimize_hyperparameters(self, X_train, y_train, cv_folds=3):
        """
        하이퍼파라미터 최적화

        Parameters:
        X_train, y_train: 학습 데이터
        cv_folds (int): 교차검증 폴드 수

        Returns:
        dict: 최적화된 파라미터
        """
        print("\n🔧 하이퍼파라미터 최적화 시작...")

        # XGBoost 파라미터 그리드
        xgb_params = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15]
        }

        # Random Forest 파라미터 그리드
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5]
        }

        param_grids = {
            'xgboost': xgb_params,
            'random_forest': rf_params
        }

        for name, model in list(self.models.items())[:2]:  # XGBoost, RF만 최적화
            if name in param_grids:
                print(f"   🔄 {name} 최적화 중...")

                grid_search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                self.best_params[name] = grid_search.best_params_
                self.models[name] = grid_search.best_estimator_

                print(f"   ✅ {name} 최적화 완료")
                print(f"      최적 파라미터: {grid_search.best_params_}")

        return self.best_params

    def create_voting_ensemble(self, weights=None):
        """
        Voting Regressor 생성

        Parameters:
        weights (list): 모델별 가중치

        Returns:
        VotingRegressor: 앙상블 모델
        """
        estimators = [(name, model) for name, model in self.models.items()]

        self.ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights
        )

        print("✅ Voting Regressor 생성 완료")
        print(f"   - 구성 모델: {len(estimators)}개")
        if weights:
            print(f"   - 가중치: {weights}")

        return self.ensemble

    def train_models(self, X_train, y_train):
        """
        모든 모델 학습

        Parameters:
        X_train, y_train: 학습 데이터

        Returns:
        dict: 학습된 모델들
        """
        print("\n🚀 모델 학습 시작...")

        # 개별 모델 학습
        for name, model in self.models.items():
            print(f"   🔄 {name} 학습 중...")
            model.fit(X_train, y_train)
            print(f"   ✅ {name} 학습 완료")

        # 앙상블 모델 학습
        if self.ensemble is not None:
            print("   🔄 앙상블 모델 학습 중...")
            self.ensemble.fit(X_train, y_train)
            print("   ✅ 앙상블 모델 학습 완료")

        print("🎯 모든 모델 학습 완료!")

        return self.models

    def evaluate_models(self, X_test, y_test):
        """
        모델 성능 평가

        Parameters:
        X_test, y_test: 테스트 데이터

        Returns:
        dict: 평가 결과
        """
        results = {}

        print("\n📊 모델 성능 평가 시작...")

        # 개별 모델 평가
        for name, model in self.models.items():
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': np.sqrt(mse),
                'R²': r2,
                'predictions': predictions
            }

            print(f"   ✅ {name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # 앙상블 모델 평가
        if self.ensemble is not None:
            ensemble_pred = self.ensemble.predict(X_test)
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)

            results['ensemble'] = {
                'MSE': ensemble_mse,
                'MAE': ensemble_mae,
                'RMSE': np.sqrt(ensemble_mse),
                'R²': ensemble_r2,
                'predictions': ensemble_pred
            }

            print(f"   ✅ ensemble: MSE={ensemble_mse:.4f}, MAE={ensemble_mae:.4f}, R²={ensemble_r2:.4f}")

        print("📊 모델 성능 평가 완료!")

        return results

    def cross_validate_models(self, X, y, cv_folds=5):
        """
        교차검증 수행

        Parameters:
        X, y: 전체 데이터
        cv_folds (int): 교차검증 폴드 수

        Returns:
        dict: 교차검증 결과
        """
        print(f"\n🔄 {cv_folds}-fold 교차검증 시작...")

        cv_results = {}

        for name, model in self.models.items():
            scores = cross_val_score(
                model, X, y,
                cv=cv_folds,
                scoring='neg_mean_squared_error'
            )

            cv_results[name] = {
                'mean_score': -scores.mean(),
                'std_score': scores.std(),
                'scores': -scores
            }

            print(f"   {name}: {-scores.mean():.4f} (±{scores.std():.4f})")

        # 앙상블 교차검증
        if self.ensemble is not None:
            ensemble_scores = cross_val_score(
                self.ensemble, X, y,
                cv=cv_folds,
                scoring='neg_mean_squared_error'
            )

            cv_results['ensemble'] = {
                'mean_score': -ensemble_scores.mean(),
                'std_score': ensemble_scores.std(),
                'scores': -ensemble_scores
            }

            print(f"   ensemble: {-ensemble_scores.mean():.4f} (±{ensemble_scores.std():.4f})")

        print("✅ 교차검증 완료!")

        return cv_results

    def plot_model_comparison(self, results, save_path='../outputs/ensemble_comparison.png'):
        """
        모델 성능 비교 시각화

        Parameters:
        results (dict): 평가 결과
        save_path (str): 저장 경로
        """
        # 메트릭 추출
        models = list(results.keys())
        metrics = ['MSE', 'MAE', 'RMSE', 'R²']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]

            bars = axes[i].bar(models, values, alpha=0.7, color=plt.cm.Set3(np.arange(len(models))))
            axes[i].set_title(f'{metric} Performance Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)

            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 성능 비교 차트 저장: {save_path}")

    def plot_prediction_comparison(self, y_test, results, save_path='../outputs/prediction_comparison.png'):
        """
        예측 결과 비교 시각화

        Parameters:
        y_test: 실제 값
        results (dict): 예측 결과
        save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        models = list(results.keys())

        for i, model_name in enumerate(models[:4]):  # 최대 4개 모델
            predictions = results[model_name]['predictions']
            r2 = results[model_name]['R²']

            axes[i].scatter(y_test, predictions, alpha=0.6, s=30)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'{model_name} (R² = {r2:.3f})')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 예측 비교 차트 저장: {save_path}")

    def get_feature_importance(self, feature_names):
        """
        특성 중요도 추출

        Parameters:
        feature_names (list): 특성명 리스트

        Returns:
        dict: 모델별 특성 중요도
        """
        importance_dict = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(feature_names, model.feature_importances_))

        return importance_dict

    def plot_feature_importance(self, feature_names, save_path='../outputs/feature_importance.png'):
        """
        특성 중요도 시각화

        Parameters:
        feature_names (list): 특성명 리스트
        save_path (str): 저장 경로
        """
        importance_dict = self.get_feature_importance(feature_names)

        if not importance_dict:
            print("⚠️ 특성 중요도를 계산할 수 있는 모델이 없습니다.")
            return

        fig, axes = plt.subplots(1, len(importance_dict), figsize=(5*len(importance_dict), 6))

        if len(importance_dict) == 1:
            axes = [axes]

        for i, (model_name, importance) in enumerate(importance_dict.items()):
            features = list(importance.keys())
            values = list(importance.values())

            # 중요도 순으로 정렬
            sorted_idx = np.argsort(values)
            features = [features[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]

            axes[i].barh(features, values, alpha=0.7)
            axes[i].set_title(f'{model_name} Feature Importance')
            axes[i].set_xlabel('Importance')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 특성 중요도 차트 저장: {save_path}")

def main():
    """메인 실행 함수"""
    print("🚀 앙상블 모델 구현 시작")
    print("="*60)

    # 앙상블 구현 객체 생성
    ensemble_impl = EnsembleModelImplementation()

    # 1. 데이터 생성
    print("\n📋 1단계: 시뮬레이션 데이터 생성")
    X, y, feature_names = ensemble_impl.generate_policy_data(n_samples=1500, n_features=8)

    # 2. 데이터 분할
    print("\n⚙️ 2단계: 데이터 분할")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   - 학습 데이터: {X_train.shape}")
    print(f"   - 테스트 데이터: {X_test.shape}")

    # 3. 기본 모델 생성
    print("\n🤖 3단계: 기본 모델 생성")
    ensemble_impl.create_base_models()

    # 4. 하이퍼파라미터 최적화
    print("\n🔧 4단계: 하이퍼파라미터 최적화")
    ensemble_impl.optimize_hyperparameters(X_train, y_train, cv_folds=3)

    # 5. 앙상블 모델 생성
    print("\n🎯 5단계: 앙상블 모델 생성")
    ensemble_impl.create_voting_ensemble()

    # 6. 모델 학습
    print("\n🚀 6단계: 모델 학습")
    ensemble_impl.train_models(X_train, y_train)

    # 7. 교차검증
    print("\n🔄 7단계: 교차검증")
    cv_results = ensemble_impl.cross_validate_models(X, y, cv_folds=5)

    # 8. 테스트 데이터 평가
    print("\n📊 8단계: 테스트 데이터 평가")
    test_results = ensemble_impl.evaluate_models(X_test, y_test)

    # 9. 결과 시각화
    print("\n📈 9단계: 결과 시각화")
    ensemble_impl.plot_model_comparison(test_results)
    ensemble_impl.plot_prediction_comparison(y_test, test_results)
    ensemble_impl.plot_feature_importance(feature_names)

    # 10. 최종 요약
    print("\n" + "="*60)
    print("📊 앙상블 모델 구현 완료!")
    print("="*60)

    best_model = min(test_results.keys(), key=lambda x: test_results[x]['MSE'])
    print(f"🏆 최고 성능 모델: {best_model}")
    print(f"   - MSE: {test_results[best_model]['MSE']:.4f}")
    print(f"   - R²: {test_results[best_model]['R²']:.4f}")

    if 'ensemble' in test_results:
        improvement = (test_results[best_model]['MSE'] - test_results['ensemble']['MSE']) / test_results[best_model]['MSE'] * 100
        if improvement > 0:
            print(f"🎯 앙상블로 인한 개선율: {improvement:.1f}%")

    print("\n📁 상세 결과는 practice/chapter05/outputs/ 폴더에서 확인하세요.")

if __name__ == "__main__":
    main()