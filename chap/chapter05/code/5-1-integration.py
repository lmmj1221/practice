#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제5장: 머신러닝과 딥러닝 통합 파이프라인 구현
ML/DL 통합 파이프라인 구축 예제
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

class MLDLIntegrationPipeline:
    """머신러닝과 딥러닝 통합 파이프라인 클래스"""

    def __init__(self, config=None):
        """
        파이프라인 초기화

        Parameters:
        config (dict): 파이프라인 설정 딕셔너리
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.ml_models = {}
        self.dl_model = None
        self.ensemble_model = None
        self.feature_names = None

    def generate_sample_data(self, n_samples=1000, n_features=5, noise=0.1):
        """
        시뮬레이션 정책 데이터 생성
        ※ 본 데이터는 교육 목적의 시뮬레이션 데이터입니다

        Parameters:
        n_samples (int): 샘플 수
        n_features (int): 특성 수
        noise (float): 노이즈 레벨

        Returns:
        tuple: (X, y) 특성과 타겟 데이터
        """
        np.random.seed(42)

        # 정책 관련 특성 생성 (정규화된 값)
        X = np.random.randn(n_samples, n_features)

        # 비선형 관계를 가진 타겟 변수 생성
        y = (2 * X[:, 0] +
             1.5 * X[:, 1] ** 2 +
             0.8 * X[:, 2] * X[:, 3] +
             0.5 * np.sin(X[:, 4]) +
             noise * np.random.randn(n_samples))

        # DataFrame으로 변환
        feature_names = [f'정책변수_{i+1}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)

        self.feature_names = feature_names

        print(f"✅ 시뮬레이션 데이터 생성 완료: {n_samples}개 샘플, {n_features}개 특성")
        return X_df, y

    def preprocess_data(self, X, y):
        """
        데이터 전처리 수행

        Parameters:
        X (DataFrame): 입력 특성
        y (array): 타겟 변수

        Returns:
        tuple: 전처리된 학습/테스트 데이터
        """
        # 결측값 처리
        if X.isnull().sum().sum() > 0:
            X_processed = X.fillna(X.median(numeric_only=True))
            print("📋 결측값 중위수로 대체")
        else:
            X_processed = X.copy()

        # 범주형 변수 인코딩 (만약 있다면)
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col])
            print(f"📋 {col} 범주형 변수 인코딩 완료")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        # 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # DataFrame으로 변환 (특성명 유지)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_processed.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_processed.columns)

        print("✅ 데이터 전처리 완료")
        print(f"   - 학습 데이터: {X_train_scaled.shape}")
        print(f"   - 테스트 데이터: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_ml_models(self, X_train, y_train):
        """
        머신러닝 모델 구축 및 학습

        Parameters:
        X_train (DataFrame): 학습용 특성 데이터
        y_train (array): 학습용 타겟 데이터

        Returns:
        dict: 학습된 머신러닝 모델들
        """
        # Random Forest 모델
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.ml_models['random_forest'] = rf_model

        print("✅ Random Forest 모델 학습 완료")

        return self.ml_models

    def build_dl_model(self, input_shape, sequence_length=10):
        """
        딥러닝 모델 구축

        Parameters:
        input_shape (int): 입력 특성 수
        sequence_length (int): 시퀀스 길이 (LSTM용)

        Returns:
        Model: 구축된 딥러닝 모델
        """
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        self.dl_model = model

        print("✅ 딥러닝 모델 구축 완료")
        print(f"   - 총 파라미터 수: {model.count_params():,}")

        return model

    def train_dl_model(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        딥러닝 모델 학습

        Parameters:
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        epochs (int): 학습 에포크 수

        Returns:
        History: 학습 이력
        """
        # 조기 종료 콜백
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # 모델 학습
        history = self.dl_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        print("✅ 딥러닝 모델 학습 완료")

        return history

    def create_ensemble(self):
        """
        앙상블 모델 생성

        Returns:
        VotingRegressor: 앙상블 모델
        """
        if not self.ml_models or self.dl_model is None:
            raise ValueError("먼저 개별 모델들을 학습해주세요")

        # ML 모델들을 위한 VotingRegressor
        ml_estimators = [(name, model) for name, model in self.ml_models.items()]

        self.ensemble_model = VotingRegressor(ml_estimators)

        print("✅ 앙상블 모델 생성 완료")

        return self.ensemble_model

    def evaluate_models(self, X_test, y_test):
        """
        모든 모델의 성능 평가

        Parameters:
        X_test (DataFrame): 테스트 특성 데이터
        y_test (array): 테스트 타겟 데이터

        Returns:
        dict: 평가 결과
        """
        results = {}

        # ML 모델 평가
        for name, model in self.ml_models.items():
            predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': np.sqrt(mse),
                'R²': r2
            }

        # DL 모델 평가
        if self.dl_model is not None:
            dl_predictions = self.dl_model.predict(X_test, verbose=0)
            dl_mse = mean_squared_error(y_test, dl_predictions)

            results['deep_learning'] = {
                'MSE': dl_mse,
                'MAE': mean_absolute_error(y_test, dl_predictions),
                'RMSE': np.sqrt(dl_mse),
                'R²': r2_score(y_test, dl_predictions)
            }

        # 앙상블 모델 평가 (ML 모델들만)
        if self.ensemble_model is not None:
            ensemble_pred = self.ensemble_model.predict(X_test)
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)

            results['ensemble'] = {
                'MSE': ensemble_mse,
                'MAE': mean_absolute_error(y_test, ensemble_pred),
                'RMSE': np.sqrt(ensemble_mse),
                'R²': r2_score(y_test, ensemble_pred)
            }

        print("📊 모든 모델 성능 평가 완료")

        return results

    def print_results(self, results):
        """
        평가 결과 출력

        Parameters:
        results (dict): 평가 결과 딕셔너리
        """
        print("\n" + "="*60)
        print("📊 모델 성능 비교 결과")
        print("="*60)

        for model_name, metrics in results.items():
            print(f"\n🔹 {model_name.upper()} 모델:")
            print(f"  MSE:  {metrics['MSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  R²:   {metrics['R²']:.4f}")

    def plot_results(self, results, save_path='../outputs/model_comparison.png'):
        """
        결과 시각화

        Parameters:
        results (dict): 평가 결과
        save_path (str): 저장 경로
        """
        metrics = ['MSE', 'MAE', 'RMSE', 'R²']
        models = list(results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]

            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric} 비교')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)

            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 결과 시각화 저장: {save_path}")

def main():
    """메인 실행 함수"""
    print("🚀 머신러닝과 딥러닝 통합 파이프라인 시작")
    print("="*60)

    # 파이프라인 초기화
    pipeline = MLDLIntegrationPipeline()

    # 1. 데이터 생성
    print("\n📋 1단계: 데이터 생성")
    X, y = pipeline.generate_sample_data(n_samples=1000, n_features=5)

    # 2. 데이터 전처리
    print("\n⚙️ 2단계: 데이터 전처리")
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(X, y)

    # 검증 데이터 분할
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 3. 머신러닝 모델 구축
    print("\n🤖 3단계: 머신러닝 모델 구축")
    pipeline.build_ml_models(X_train, y_train)

    # 4. 딥러닝 모델 구축 및 학습
    print("\n🧠 4단계: 딥러닝 모델 구축 및 학습")
    pipeline.build_dl_model(X_train.shape[1])
    pipeline.train_dl_model(X_train_split, y_train_split, X_val, y_val, epochs=50)

    # 5. 앙상블 모델 생성
    print("\n🎯 5단계: 앙상블 모델 생성")
    pipeline.create_ensemble()
    pipeline.ensemble_model.fit(X_train, y_train)

    # 6. 모델 평가
    print("\n📊 6단계: 모델 성능 평가")
    results = pipeline.evaluate_models(X_test, y_test)

    # 7. 결과 출력 및 시각화
    pipeline.print_results(results)
    pipeline.plot_results(results)

    print("\n✅ 통합 파이프라인 실행 완료!")
    print("📁 상세 결과는 practice/chapter05/outputs/ 폴더에서 확인하세요.")

if __name__ == "__main__":
    main()