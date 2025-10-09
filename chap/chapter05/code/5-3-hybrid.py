#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제5장: LSTM-Transformer 하이브리드 시계열 모델
정책 시계열 데이터를 위한 하이브리드 딥러닝 모델 구현
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add,
    Bidirectional, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

class LSTMTransformerHybrid:
    """LSTM-Transformer 하이브리드 모델 클래스"""

    def __init__(self, sequence_length=20, num_features=5, lstm_units=64, attention_heads=8, d_model=128):
        """
        하이브리드 모델 초기화

        Parameters:
        sequence_length (int): 시퀀스 길이
        num_features (int): 특성 수
        lstm_units (int): LSTM 유닛 수
        attention_heads (int): 어텐션 헤드 수
        d_model (int): 모델 차원
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.d_model = d_model
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def generate_policy_timeseries(self, n_samples=2000, noise_level=0.1):
        """
        정책 관련 시계열 데이터 생성
        ※ 본 데이터는 교육 목적의 시뮬레이션 데이터입니다

        Parameters:
        n_samples (int): 생성할 시계열 길이
        noise_level (float): 노이즈 레벨

        Returns:
        DataFrame: 생성된 시계열 데이터
        """
        np.random.seed(42)

        # 시간 인덱스 생성
        time_idx = np.arange(n_samples)

        # 기본 트렌드와 계절성 패턴
        trend = 0.001 * time_idx + 2.0
        seasonal = 0.5 * np.sin(2 * np.pi * time_idx / 12) + 0.3 * np.sin(2 * np.pi * time_idx / 52)

        # 정책 관련 특성들 생성
        features = {}

        # 1. 경제성장률 (트렌드 + 계절성 + 순환 패턴)
        features['경제성장률'] = (trend +
                               seasonal +
                               0.2 * np.sin(2 * np.pi * time_idx / 20) +
                               noise_level * np.random.randn(n_samples))

        # 2. 실업률 (역순환 패턴)
        features['실업률'] = (5.0 - 0.5 * features['경제성장률'] +
                            0.3 * np.sin(2 * np.pi * time_idx / 30 + np.pi) +
                            noise_level * np.random.randn(n_samples))

        # 3. 인플레이션율 (지연된 경제성장률 반응)
        inflation_base = np.roll(features['경제성장률'], 3) * 0.4 + 2.0
        features['인플레이션율'] = (inflation_base +
                               0.2 * np.sin(2 * np.pi * time_idx / 8) +
                               noise_level * np.random.randn(n_samples))

        # 4. 정부지출 (정책 개입 시뮬레이션)
        gov_spending = 20.0 + 0.01 * time_idx
        # 정책 충격 추가 (특정 시점에서 큰 변화)
        shock_points = [500, 1000, 1500]
        for shock in shock_points:
            if shock < n_samples:
                gov_spending[shock:shock+50] += 5.0 * np.exp(-np.arange(50) / 10)

        features['정부지출비율'] = gov_spending + noise_level * np.random.randn(n_samples)

        # 5. 정책효과지수 (복합 지표)
        features['정책효과지수'] = (0.4 * features['경제성장률'] -
                               0.2 * features['실업률'] -
                               0.1 * features['인플레이션율'] +
                               0.02 * features['정부지출비율'] +
                               noise_level * np.random.randn(n_samples))

        # DataFrame 생성
        df = pd.DataFrame(features)
        df.index = pd.date_range(start='2020-01-01', periods=n_samples, freq='W')

        print(f"✅ 정책 시계열 데이터 생성 완료")
        print(f"   - 시계열 길이: {n_samples}")
        print(f"   - 특성 수: {len(features)}")
        print(f"   - 기간: {df.index[0]} ~ {df.index[-1]}")

        return df

    def create_sequences(self, data, target_col='정책효과지수'):
        """
        시계열 데이터를 시퀀스로 변환

        Parameters:
        data (DataFrame): 시계열 데이터
        target_col (str): 타겟 컬럼명

        Returns:
        tuple: (X, y) 시퀀스 데이터
        """
        # 데이터 정규화
        scaled_data = self.scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

        # 타겟 컬럼 인덱스
        target_idx = data.columns.get_loc(target_col)

        X, y = [], []

        for i in range(self.sequence_length, len(scaled_df)):
            # 입력 시퀀스 (모든 특성)
            X.append(scaled_data[i-self.sequence_length:i])
            # 타겟 (다음 시점의 타겟 값)
            y.append(scaled_data[i, target_idx])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ 시퀀스 데이터 생성 완료")
        print(f"   - 입력 형태: {X.shape}")
        print(f"   - 타겟 형태: {y.shape}")

        return X, y

    def build_hybrid_model(self):
        """
        LSTM-Transformer 하이브리드 모델 구축

        Returns:
        Model: 구축된 모델
        """
        # 입력 층
        inputs = Input(shape=(self.sequence_length, self.num_features), name='input_sequence')

        # LSTM 브랜치
        lstm_out = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
            name='bidirectional_lstm'
        )(inputs)

        lstm_out = LayerNormalization(name='lstm_norm')(lstm_out)

        # Transformer 브랜치
        # Multi-Head Attention
        attention_out = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.d_model // self.attention_heads,
            name='multi_head_attention'
        )(inputs, inputs)

        # Add & Norm
        attention_out = Add(name='attention_add')([inputs, attention_out])
        attention_out = LayerNormalization(name='attention_norm')(attention_out)

        # Feed Forward Network
        ffn_out = Dense(self.d_model * 2, activation='relu', name='ffn_1')(attention_out)
        ffn_out = Dropout(0.1, name='ffn_dropout_1')(ffn_out)
        ffn_out = Dense(self.num_features, name='ffn_2')(ffn_out)
        ffn_out = Dropout(0.1, name='ffn_dropout_2')(ffn_out)

        # Add & Norm
        transformer_out = Add(name='ffn_add')([attention_out, ffn_out])
        transformer_out = LayerNormalization(name='transformer_norm')(transformer_out)

        # 브랜치 결합
        # LSTM 출력을 Transformer 차원에 맞게 조정
        lstm_projected = Dense(self.num_features, name='lstm_projection')(lstm_out)

        # 두 브랜치 결합
        combined = Add(name='branch_combination')([lstm_projected, transformer_out])

        # Global Average Pooling
        pooled = GlobalAveragePooling1D(name='global_avg_pool')(combined)

        # Dense 층들
        dense_out = Dense(128, activation='relu', name='dense_1')(pooled)
        dense_out = Dropout(0.3, name='dense_dropout_1')(dense_out)

        dense_out = Dense(64, activation='relu', name='dense_2')(dense_out)
        dense_out = Dropout(0.3, name='dense_dropout_2')(dense_out)

        dense_out = Dense(32, activation='relu', name='dense_3')(dense_out)
        dense_out = Dropout(0.2, name='dense_dropout_3')(dense_out)

        # 출력 층
        outputs = Dense(1, name='output')(dense_out)

        # 모델 생성
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Transformer_Hybrid')

        # 컴파일
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("✅ LSTM-Transformer 하이브리드 모델 구축 완료")
        print(f"   - 총 파라미터 수: {self.model.count_params():,}")

        return self.model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        모델 학습

        Parameters:
        X_train, y_train: 학습 데이터
        X_val, y_val: 검증 데이터
        epochs (int): 에포크 수
        batch_size (int): 배치 크기

        Returns:
        History: 학습 이력
        """
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]

        print("🚀 하이브리드 모델 학습 시작...")

        # 모델 학습
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("✅ 하이브리드 모델 학습 완료")

        return self.history

    def evaluate_model(self, X_test, y_test):
        """
        모델 평가

        Parameters:
        X_test, y_test: 테스트 데이터

        Returns:
        dict: 평가 결과
        """
        # 예측
        y_pred = self.model.predict(X_test, verbose=0)

        # 평가 지표 계산
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'predictions': y_pred.flatten()
        }

        print(f"📊 모델 평가 결과:")
        print(f"   - MSE: {mse:.6f}")
        print(f"   - MAE: {mae:.6f}")
        print(f"   - RMSE: {rmse:.6f}")
        print(f"   - R²: {r2:.6f}")

        return results

    def plot_training_history(self, save_path='practice/chapter05/outputs/training_history.png'):
        """
        학습 이력 시각화

        Parameters:
        save_path (str): 저장 경로
        """
        if self.history is None:
            print("⚠️ 학습 이력이 없습니다.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 그래프
        ax1.plot(self.history.history['loss'], label='학습 Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='검증 Loss', linewidth=2)
        ax1.set_title('모델 Loss 변화')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE 그래프
        ax2.plot(self.history.history['mae'], label='학습 MAE', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='검증 MAE', linewidth=2)
        ax2.set_title('모델 MAE 변화')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 학습 이력 저장: {save_path}")

    def plot_predictions(self, y_true, y_pred, save_path='practice/chapter05/outputs/hybrid_predictions.png'):
        """
        예측 결과 시각화

        Parameters:
        y_true: 실제 값
        y_pred: 예측 값
        save_path (str): 저장 경로
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 시계열 비교
        time_steps = range(len(y_true))
        ax1.plot(time_steps, y_true, label='실제 값', alpha=0.8, linewidth=1.5)
        ax1.plot(time_steps, y_pred, label='예측 값', alpha=0.8, linewidth=1.5)
        ax1.set_title('시계열 예측 비교')
        ax1.set_xlabel('시간 단계')
        ax1.set_ylabel('정규화된 값')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 산점도
        ax2.scatter(y_true, y_pred, alpha=0.6, s=20)
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax2.set_xlabel('실제 값')
        ax2.set_ylabel('예측 값')
        ax2.set_title('예측 정확도 산점도')
        ax2.grid(True, alpha=0.3)

        # R² 값 표시
        r2 = r2_score(y_true, y_pred)
        ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 예측 결과 저장: {save_path}")

    def get_attention_weights(self, X_sample):
        """
        어텐션 가중치 추출

        Parameters:
        X_sample: 입력 샘플

        Returns:
        array: 어텐션 가중치
        """
        # 어텐션 층의 출력을 얻기 위한 중간 모델 생성
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, MultiHeadAttention):
                attention_layer = layer
                break

        if attention_layer is None:
            print("⚠️ 어텐션 층을 찾을 수 없습니다.")
            return None

        # 중간 모델 생성 (어텐션 출력까지)
        attention_model = Model(
            inputs=self.model.input,
            outputs=attention_layer.output
        )

        # 어텐션 출력 계산
        attention_output = attention_model.predict(X_sample, verbose=0)

        return attention_output

    def print_model_summary(self):
        """모델 구조 요약 출력"""
        if self.model is None:
            print("⚠️ 모델이 구축되지 않았습니다.")
            return

        print("\n🏗️ 모델 구조 요약:")
        print("="*50)
        self.model.summary()

def main():
    """메인 실행 함수"""
    print("🚀 LSTM-Transformer 하이브리드 모델 구현 시작")
    print("="*60)

    # 하이브리드 모델 객체 생성
    sequence_length = 20
    hybrid_model = LSTMTransformerHybrid(
        sequence_length=sequence_length,
        num_features=5,
        lstm_units=64,
        attention_heads=8,
        d_model=128
    )

    # 1. 시계열 데이터 생성
    print("\n📋 1단계: 정책 시계열 데이터 생성")
    data = hybrid_model.generate_policy_timeseries(n_samples=2000, noise_level=0.1)

    # 데이터 확인
    print(f"\n📊 데이터 요약:")
    print(data.describe())

    # 2. 시퀀스 데이터 생성
    print("\n⚙️ 2단계: 시퀀스 데이터 생성")
    X, y = hybrid_model.create_sequences(data, target_col='정책효과지수')

    # 3. 데이터 분할
    print("\n📂 3단계: 데이터 분할")
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"   - 학습 데이터: {X_train.shape}")
    print(f"   - 검증 데이터: {X_val.shape}")
    print(f"   - 테스트 데이터: {X_test.shape}")

    # 4. 모델 구축
    print("\n🏗️ 4단계: 하이브리드 모델 구축")
    hybrid_model.build_hybrid_model()
    hybrid_model.print_model_summary()

    # 5. 모델 학습
    print("\n🚀 5단계: 모델 학습")
    hybrid_model.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )

    # 6. 모델 평가
    print("\n📊 6단계: 모델 평가")
    results = hybrid_model.evaluate_model(X_test, y_test)

    # 7. 결과 시각화
    print("\n📈 7단계: 결과 시각화")
    hybrid_model.plot_training_history()
    hybrid_model.plot_predictions(y_test, results['predictions'])

    # 8. 어텐션 분석 (샘플)
    print("\n🔍 8단계: 어텐션 분석")
    sample_input = X_test[:5]  # 5개 샘플
    attention_weights = hybrid_model.get_attention_weights(sample_input)

    if attention_weights is not None:
        print(f"   - 어텐션 출력 형태: {attention_weights.shape}")
        print("   - 어텐션 메커니즘이 시계열 패턴을 성공적으로 포착했습니다.")

    # 9. 최종 요약
    print("\n" + "="*60)
    print("🎯 LSTM-Transformer 하이브리드 모델 구현 완료!")
    print("="*60)
    print(f"📊 최종 성능:")
    print(f"   - RMSE: {results['RMSE']:.6f}")
    print(f"   - MAE: {results['MAE']:.6f}")
    print(f"   - R²: {results['R²']:.6f}")

    print(f"\n🏗️ 모델 구조:")
    print(f"   - 총 파라미터: {hybrid_model.model.count_params():,}")
    print(f"   - LSTM 유닛: {hybrid_model.lstm_units}")
    print(f"   - 어텐션 헤드: {hybrid_model.attention_heads}")

    print("\n📁 상세 결과는 practice/chapter05/outputs/ 폴더에서 확인하세요.")

if __name__ == "__main__":
    main()