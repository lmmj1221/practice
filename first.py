"""
제3장: 딥러닝 기초와 정책 시계열 예측
Complete implementation with all sections
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 출력 디렉토리가 없으면 생성
os.makedirs('output', exist_ok=True)

# =====================================================
# Section 3.1: 딥러닝 기초와 신경망
# =====================================================

print("=" * 60)
print("Section 3.1: 딥러닝 기초와 신경망")
print("=" * 60)

# 정책 효과 분류를 위한 간단한 MLP
def create_mlp_model(input_dim, hidden_units=[64, 32], output_dim=1):
    """
    정책 영향 예측을 위한 다층 퍼셉트론 생성

    인자:
        input_dim: 입력 특징의 개수
        hidden_units: 각 은닉층의 뉴런 수 리스트
        output_dim: 출력 클래스/값의 개수
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # ReLU 활성화 함수를 가진 은닉층 추가
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))
    
    # 출력층
    model.add(keras.layers.Dense(output_dim, activation='sigmoid'))
    
    return model

# 정책 예측을 위한 손실 함수
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def mape_loss(y_true, y_pred):
    epsilon = 1e-7  # 0으로 나누는 것을 방지
    percentage_error = tf.abs((y_true - y_pred) / (y_true + epsilon))
    return tf.reduce_mean(percentage_error) * 100

def policy_aware_loss(y_true, y_pred, policy_phase):
    """
    정책 단계에 따라 오류에 다른 가중치를 부여하는 커스텀 손실 함수
    """
    base_loss = mse_loss(y_true, y_pred)
    
    # 정책 시행 기간 동안 더 높은 페널티 부여
    policy_weight = tf.where(policy_phase > 0, 2.0, 1.0)
    weighted_loss = base_loss * policy_weight
    
    return tf.reduce_mean(weighted_loss)

# Lion 옵티마이저 구현
class Lion(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.99, 
                 weight_decay=0.0, name="Lion", **kwargs):
        super().__init__(name=name, **kwargs)
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._weight_decay = weight_decay
        
    def build(self, var_list):
        super().build(var_list)
        self._momentums = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(var, "momentum")
            )

# 특징 엔지니어링 함수
def create_temporal_features(df):
    """전력 수요를 위한 수작업 특징 생성"""
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 한국 공휴일 (간소화된 목록)
    korean_holidays = pd.to_datetime([
        '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11',
        '2024-03-01', '2024-05-05', '2024-05-15', '2024-06-06',
        '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18',
        '2024-10-03', '2024-10-09', '2024-12-25'
    ])
    df['is_holiday'] = df['timestamp'].isin(korean_holidays).astype(int)
    
    # 순환 인코딩
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

# 특징 추출을 위한 오토인코더
class AutoEncoder(keras.Model):
    """딥 오토인코더를 사용한 자동 특징 추출"""
    def __init__(self, input_dim, encoding_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(encoding_dim, activation='relu')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(input_dim, activation='sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def preprocess_policy_data(df):
    """
    특별한 고려사항을 포함한 정책 시계열 데이터 전처리
    """
    # 전진 채우기로 결측값 처리 (정책 연속성 가정)
    df = df.fillna(method='ffill')
    
    # 로버스트 스케일링을 사용한 정규화 (이상치에 강함)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # 정책 개입 지표 분리
    policy_cols = ['policy_intervention', 'policy_phase']
    feature_cols = [col for col in df.columns if col not in policy_cols and col != 'timestamp']
    
    # 연속 특징 스케일링
    if feature_cols:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 지연된 정책 효과를 위한 시차 특징 생성
    if 'demand_mw' in df.columns:
        for lag in [1, 7, 30]:  # 1일, 1주일, 1개월 시차
            df[f'demand_lag_{lag}'] = df['demand_mw'].shift(lag)
        
        # 이동 통계 생성
        for window in [24, 168]:  # 일별 및 주별 윈도우
            df[f'demand_ma_{window}'] = df['demand_mw'].rolling(window).mean()
            df[f'demand_std_{window}'] = df['demand_mw'].rolling(window).std()
    
    return df

print("\n✅ Deep learning components initialized")

# =====================================================
# Section 3.2: 시계열 데이터와 정책 분석
# =====================================================

print("\n" + "=" * 60)
print("Section 3.2: 시계열 데이터와 정책 분석")
print("=" * 60)

def detect_intervention_points(ts_data, threshold=3):
    """
    통계적 방법을 사용하여 정책 개입 시점 감지
    """
    # 이동 통계 계산
    window = 30
    rolling_mean = ts_data.rolling(window).mean()
    rolling_std = ts_data.rolling(window).std()
    
    # 이상치 감지를 위한 Z-점수
    z_scores = np.abs(stats.zscore(ts_data.dropna()))
    
    # CUSUM을 사용한 구조적 변화 감지
    def cusum(data):
        mean = np.mean(data)
        cumsum = np.cumsum(data - mean)
        return cumsum
    
    cusum_values = cusum(ts_data.dropna())
    
    # 개입 시점 식별
    interventions = []
    for i in range(len(z_scores)):
        if z_scores[i] > threshold:
            interventions.append({
                'index': i,
                'value': ts_data.iloc[i] if hasattr(ts_data, 'iloc') else ts_data[i],
                'z_score': z_scores[i],
                'type': 'anomaly'
            })
    
    return interventions

# 정상성 테스트
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, significance_level=0.05):
    """
    ADF 테스트를 사용하여 시계열의 정상성 검사
    """
    result = adfuller(timeseries.dropna())
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.3f}')
    
    if result[1] <= significance_level:
        print("✅ Series is stationary")
        return True
    else:
        print("❌ Series is non-stationary, differencing needed")
        return False

def make_stationary(ts_data):
    """
    비정상 시계열을 정상 시계열로 변환
    """
    # 1차 차분
    diff_1 = ts_data.diff().dropna()
    
    if test_stationarity(diff_1):
        return diff_1, 1
    
    # 필요시 2차 차분
    diff_2 = diff_1.diff().dropna()
    if test_stationarity(diff_2):
        return diff_2, 2
    
    # 로그 변환 + 차분
    log_diff = np.log(ts_data[ts_data > 0]).diff().dropna()
    return log_diff, 'log_diff'

# 외부 변수를 위한 다변량 LSTM
class MultivariateLSTM(keras.Model):
    """
    정책 예측을 위한 외부 변수가 포함된 LSTM 모델
    """
    def __init__(self, n_features, n_external, lstm_units=50):
        super(MultivariateLSTM, self).__init__()
        
        # 시계열을 위한 LSTM
        self.lstm_1 = keras.layers.LSTM(lstm_units, return_sequences=True)
        self.lstm_2 = keras.layers.LSTM(lstm_units // 2)
        
        # 외부 변수를 위한 밀집층
        self.external_dense = keras.layers.Dense(32, activation='relu')
        
        # 결합층
        self.combine = keras.layers.Concatenate()
        self.output_dense = keras.layers.Dense(1)
    
    def call(self, inputs):
        ts_input, external_input = inputs
        
        # 시계열 처리
        lstm_out = self.lstm_1(ts_input)
        lstm_out = self.lstm_2(lstm_out)
        
        # 외부 변수 처리
        external_out = self.external_dense(external_input)
        
        # 결합 및 출력
        combined = self.combine([lstm_out, external_out])
        output = self.output_dense(combined)
        
        return output

def generate_counterfactual(model, data, intervention_start, intervention_end):
    """
    정책 개입 없이 반사실적 예측 생성
    """
    # 데이터 복사 및 개입 제거
    counterfactual_data = data.copy()
    counterfactual_data.loc[intervention_start:intervention_end, 'policy_intervention'] = 0
    
    # 개입이 있는 경우와 없는 경우의 예측
    with_policy = model.predict(data)
    without_policy = model.predict(counterfactual_data)
    
    # 정책 효과 계산
    policy_effect = with_policy - without_policy
    
    return {
        'with_policy': with_policy,
        'without_policy': without_policy,
        'policy_effect': policy_effect,
        'average_effect': np.mean(policy_effect[intervention_start:intervention_end])
    }

# 동적 처리 효과 추정
class DynamicTreatmentEffect(keras.Model):
    """
    신경망을 사용하여 시간에 따라 변하는 처리 효과 추정
    """
    def __init__(self, hidden_dim=64):
        super(DynamicTreatmentEffect, self).__init__()
        
        # 공유 표현층
        self.shared = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim // 2, activation='relu')
        ])
        
        # 처리별 헤드
        self.control_head = keras.layers.Dense(1)
        self.treatment_head = keras.layers.Dense(1)
    
    def call(self, inputs, treatment):
        # 공유 표현
        shared_rep = self.shared(inputs)
        
        # 처리별 예측
        if treatment == 0:
            return self.control_head(shared_rep)
        else:
            return self.treatment_head(shared_rep)
    
    def estimate_effect(self, inputs):
        """개별 처리 효과 추정"""
        y0 = self.call(inputs, treatment=0)
        y1 = self.call(inputs, treatment=1)
        return y1 - y0

print("\n✅ Time series analysis tools initialized")

# =====================================================
# Section 3.3: RNN의 구조와 한계
# =====================================================

print("\n" + "=" * 60)
print("Section 3.3: RNN의 구조와 한계")
print("=" * 60)

class SimpleRNN(keras.Model):
    """
    메커니즘 이해를 위한 간단한 RNN 구현
    """
    def __init__(self, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 가중치
        self.W_xh = keras.layers.Dense(hidden_size, use_bias=False)
        self.W_hh = keras.layers.Dense(hidden_size, use_bias=False)
        self.W_hy = keras.layers.Dense(output_size, use_bias=False)
        
        # 편향
        self.b_h = self.add_weight(shape=(hidden_size,), initializer='zeros')
        self.b_y = self.add_weight(shape=(output_size,), initializer='zeros')
    
    def call(self, inputs, initial_hidden=None):
        batch_size, seq_len, input_size = inputs.shape
        
        if initial_hidden is None:
            hidden = tf.zeros((batch_size, self.hidden_size))
        else:
            hidden = initial_hidden
        
        outputs = []
        
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            
            # 은닉 상태 업데이트
            hidden = tf.nn.tanh(
                self.W_xh(x_t) + self.W_hh(hidden) + self.b_h
            )
            
            # 출력 계산
            output = self.W_hy(hidden) + self.b_y
            outputs.append(output)
        
        return tf.stack(outputs, axis=1), hidden

# 전력 수요 예측을 위한 실용 RNN
def build_rnn_model(seq_length, n_features, n_outputs=24):
    """
    24시간 예측을 위한 RNN 모델 구축
    """
    model = keras.Sequential([
        keras.layers.SimpleRNN(64, return_sequences=True, 
                               input_shape=(seq_length, n_features)),
        keras.layers.Dropout(0.2),
        keras.layers.SimpleRNN(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_outputs)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def visualize_gradient_flow(model, X, y):
    """
    RNN 층을 통한 그래디언트 흐름 시각화
    """
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = tf.reduce_mean(tf.square(y - predictions))
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    gradient_norms = []
    for grad, var in zip(gradients, model.trainable_variables):
        if grad is not None:
            norm = tf.norm(grad).numpy()
            gradient_norms.append({
                'layer': var.name,
                'gradient_norm': norm,
                'vanishing': norm < 1e-5,
                'exploding': norm > 1e3
            })
    
    return gradient_norms

print("\n✅ RNN components initialized")

# =====================================================
# Section 3.4: LSTM과 GRU
# =====================================================

print("\n" + "=" * 60)
print("Section 3.4: LSTM과 GRU")
print("=" * 60)

# 정책 인식 주의 메커니즘
class PolicyAwareAttention(keras.Model):
    """
    개념적 정책 인식 주의 메커니즘
    """
    def __init__(self, hidden_size):
        super(PolicyAwareAttention, self).__init__()
        self.lstm = keras.layers.LSTM(hidden_size, return_sequences=True)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=hidden_size // 4
        )
        self.output_layer = keras.layers.Dense(1)
    
    def call(self, inputs, policy_indicators=None):
        # 표준 LSTM 인코딩
        lstm_out = self.lstm(inputs)
        
        # 주의 메커니즘 적용
        attended = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 여기에 정책 단계 가중치를 추가할 수 있음
        if policy_indicators is not None:
            # 간단한 단계 인식 가중치 예제
            phase_weights = tf.expand_dims(policy_indicators, -1)
            attended = attended * (1 + phase_weights * 0.1)
        
        output = self.output_layer(attended)
        return output

def compare_lstm_gru_efficiency():
    """
    LSTM과 GRU의 계산 효율성 비교
    """
    import time
    
    seq_length = 168  # 1주일 시간당 데이터
    n_features = 10
    batch_size = 32
    
    # 더미 데이터 생성
    X = tf.random.normal((batch_size, seq_length, n_features))
    
    # LSTM 모델
    lstm_model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(24)
    ])
    
    # GRU 모델
    gru_model = keras.Sequential([
        keras.layers.GRU(64, return_sequences=True),
        keras.layers.GRU(32),
        keras.layers.Dense(24)
    ])
    
    # LSTM 시간 측정
    start = time.time()
    for _ in range(100):
        _ = lstm_model(X)
    lstm_time = time.time() - start
    
    # GRU 시간 측정
    start = time.time()
    for _ in range(100):
        _ = gru_model(X)
    gru_time = time.time() - start
    
    print(f"LSTM time: {lstm_time:.2f}s")
    print(f"GRU time: {gru_time:.2f}s")
    print(f"GRU is {lstm_time/gru_time:.1f}x faster")
    
    # 파라미터 수 계산
    lstm_params = lstm_model.count_params()
    gru_params = gru_model.count_params()
    
    print(f"LSTM parameters: {lstm_params:,}")
    print(f"GRU parameters: {gru_params:,}")
    print(f"GRU has {(1-gru_params/lstm_params)*100:.1f}% fewer parameters")

# 간소화된 Mamba 상태 공간 모델
class MambaBlock(keras.Model):
    """
    간소화된 Mamba 상태 공간 모델 블록
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = int(self.expand * self.d_model)
        
        # 투영층
        self.in_proj = keras.layers.Dense(d_inner * 2, use_bias=False)
        self.out_proj = keras.layers.Dense(d_model, use_bias=False)
        
        # SSM 파라미터
        self.A = self.add_weight(
            shape=(d_inner, d_state),
            initializer='glorot_uniform',
            trainable=False
        )
        self.B = keras.layers.Dense(d_state, use_bias=False)
        self.C = keras.layers.Dense(d_inner, use_bias=False)
        self.D = self.add_weight(shape=(d_inner,), initializer='ones')
        
        # 이산화 파라미터
        self.delta = keras.layers.Dense(d_inner, use_bias=False)
    
    def selective_scan(self, x, delta, A, B, C, D):
        """
        하드웨어 인식 구현을 가진 선택적 스캔 알고리즘
        """
        batch, length, d_inner = x.shape
        
        # 연속 파라미터 이산화
        deltaA = tf.exp(tf.einsum('bld,dn->bldn', delta, A))
        deltaB = tf.einsum('bld,bln->bldn', delta, B)
        
        # 선택적 스캔
        states = []
        state = tf.zeros((batch, self.d_state, d_inner))
        
        for i in range(length):
            state = deltaA[:, i] * state + deltaB[:, i] * tf.expand_dims(x[:, i], 1)
            y = tf.einsum('bdn,bn->bd', state, C[:, i])
            states.append(y)
        
        return tf.stack(states, axis=1)
    
    def call(self, x):
        batch, length, _ = x.shape
        
        # 입력 투영
        x_proj = self.in_proj(x)
        x, z = tf.split(x_proj, 2, axis=-1)
        
        # SSM 분기
        delta = self.delta(x)
        B = self.B(x)
        C = self.C(x)
        
        # 선택적 스캔 적용
        y = self.selective_scan(x, delta, self.A, B, C, self.D)
        
        # 게이트 연결
        y = y * tf.nn.silu(z)
        
        # 출력 투영
        output = self.out_proj(y)
        
        return output

# 개념적 하이브리드 모델
class ConceptualHybridModel(keras.Model):
    """
    개념적 하이브리드 아키텍처 (경험적으로 검증되지 않음)
    주의 메커니즘과 상태 공간 모델의 결합
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=6):
        super(ConceptualHybridModel, self).__init__()
        
        # 이것은 개념적 예제로, 출판된 연구에 기반하지 않음
        self.attention_layer = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads
        )
        self.feedforward = keras.layers.Dense(d_model)
        
    def call(self, x):
        # 간단한 주의 메커니즘
        attended = self.attention_layer(x, x, x)
        output = self.feedforward(attended)
        return output

print("\n✅ LSTM/GRU/Mamba components initialized")
print("\nRunning efficiency comparison...")
compare_lstm_gru_efficiency()

# =====================================================
# Section 3.5: 정책 시계열 예측 실습
# =====================================================

print("\n" + "=" * 60)
print("Section 3.5: 정책 시계열 예측 실습")
print("=" * 60)

# 데이터 생성 및 저장
def generate_and_save_data(output_dir='data'):
    """
    합성 데이터를 생성하고 CSV 파일로 저장
    """
    np.random.seed(42)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 날짜 범위 생성 (2024년 전체, 시간당 데이터)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='h')

    # 기본 수요 패턴 (MW)
    base_demand = 65000
    hourly_pattern = np.array([0.7, 0.65, 0.6, 0.58, 0.57, 0.58,
                               0.65, 0.75, 0.85, 0.9, 0.92, 0.94,
                               0.93, 0.92, 0.93, 0.94, 0.95, 0.93,
                               0.9, 0.85, 0.8, 0.75, 0.72, 0.71])

    # 합성 수요 데이터 생성
    demand_data = []
    for date in dates:
        hour_factor = hourly_pattern[date.hour]
        seasonal_factor = 1.1 if date.month in [7, 8, 12, 1, 2] else 1.0
        weekend_factor = 0.85 if date.weekday() >= 5 else 1.0
        noise = np.random.normal(0, 0.02)
        demand = base_demand * hour_factor * seasonal_factor * weekend_factor * (1 + noise)
        demand_data.append(demand)

    # 수요 데이터프레임 생성
    demand_df = pd.DataFrame({
        'timestamp': dates,
        'demand_mw': demand_data,
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (365*24)) +
                      np.random.normal(0, 2, len(dates)),
        'solar_generation_mw': np.maximum(0, 5000 * np.sin(np.pi * dates.hour / 24) *
                                         (1 - 0.3 * np.random.random(len(dates)))),
        'wind_generation_mw': 2000 + 1000 * np.random.random(len(dates)),
        'is_holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
    })

    # 정책 데이터프레임 생성
    policy_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    policy_df = pd.DataFrame({
        'date': policy_dates,
        'policy_phase': np.random.choice([0, 1, 2, 3, 4], len(policy_dates),
                                        p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'policy_intervention': np.random.choice([0, 1], len(policy_dates), p=[0.8, 0.2]),
        'renewable_target': np.linspace(20, 35, len(policy_dates)),
        'carbon_price': 50000 + 10000 * np.random.random(len(policy_dates)),
        'rec_price': 80000 + 20000 * np.random.random(len(policy_dates))
    })

    # 시장 데이터프레임 생성 (월별)
    market_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    market_df = pd.DataFrame({
        'date': market_dates,
        'smp_avg': 120000 + 30000 * np.random.random(len(market_dates)),
        'demand_avg': 65000 + 5000 * np.random.random(len(market_dates)),
        'renewable_ratio': np.linspace(15, 25, len(market_dates)) + np.random.normal(0, 2, len(market_dates))
    })

    # CSV 파일로 저장
    demand_df.to_csv(os.path.join(output_dir, 'energy_demand.csv'), index=False)
    policy_df.to_csv(os.path.join(output_dir, 'renewable_policy.csv'), index=False)
    market_df.to_csv(os.path.join(output_dir, 'electricity_market.csv'), index=False)

    print(f"✅ 데이터가 생성되어 '{output_dir}' 폴더에 저장되었습니다:")
    print(f"   - energy_demand.csv: {len(demand_df)} 시간별 레코드")
    print(f"   - renewable_policy.csv: {len(policy_df)} 일별 레코드")
    print(f"   - electricity_market.csv: {len(market_df)} 월별 레코드")

    return demand_df, policy_df, market_df

# 데이터 로드 및 준비
def load_and_prepare_data(data_dir='data', generate_if_missing=True):
    """
    저장된 데이터를 로드하고 모델링을 위해 준비
    """
    # 파일 경로 설정
    demand_file = os.path.join(data_dir, 'energy_demand.csv')
    policy_file = os.path.join(data_dir, 'renewable_policy.csv')
    market_file = os.path.join(data_dir, 'electricity_market.csv')

    # 파일 존재 확인
    if not all(os.path.exists(f) for f in [demand_file, policy_file, market_file]):
        if generate_if_missing:
            print("⚠️ 데이터 파일이 없습니다. 새로운 데이터를 생성합니다...")
            demand_df, policy_df, market_df = generate_and_save_data(data_dir)
        else:
            raise FileNotFoundError(f"데이터 파일을 {data_dir} 폴더에서 찾을 수 없습니다.")
    else:
        # 데이터 로드
        print(f"📂 '{data_dir}' 폴더에서 데이터를 로드합니다...")
        demand_df = pd.read_csv(demand_file)
        demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])

        policy_df = pd.read_csv(policy_file)
        policy_df['date'] = pd.to_datetime(policy_df['date'])

        market_df = pd.read_csv(market_file)
        market_df['date'] = pd.to_datetime(market_df['date'])

        print(f"✅ 데이터가 성공적으로 로드되었습니다:")
        print(f"   - 에너지 수요: {len(demand_df)} 레코드")
        print(f"   - 정책 데이터: {len(policy_df)} 레코드")
        print(f"   - 시장 데이터: {len(market_df)} 레코드")

    # 시간적 특징 추가
    demand_df = create_temporal_features(demand_df)

    # 정책 정보 병합
    demand_df['date'] = demand_df['timestamp'].dt.date
    policy_df['date_only'] = policy_df['date'].dt.date

    demand_df = demand_df.merge(
        policy_df[['date_only', 'policy_phase', 'policy_intervention']],
        left_on='date',
        right_on='date_only',
        how='left'
    )
    demand_df.drop(['date', 'date_only'], axis=1, inplace=True)

    # 누락된 정책 값 채우기
    demand_df['policy_phase'] = demand_df['policy_phase'].fillna(0).astype(int)
    demand_df['policy_intervention'] = demand_df['policy_intervention'].fillna(0).astype(int)

    # 시차 특징 추가
    for lag in [24, 48, 168]:  # 1일, 2일, 1주일
        demand_df[f'demand_lag_{lag}'] = demand_df['demand_mw'].shift(lag)

    # 이동 통계 추가
    for window in [24, 168]:  # 일별 및 주별 윈도우
        demand_df[f'demand_ma_{window}'] = demand_df['demand_mw'].rolling(window).mean()
        demand_df[f'demand_std_{window}'] = demand_df['demand_mw'].rolling(window).std()

    # NaN 값 제거
    demand_df = demand_df.dropna()

    print(f"✅ 전처리 완료: {demand_df.shape[0]} 샘플, {demand_df.shape[1]} 특징")

    return demand_df, policy_df


# LSTM 훈련을 위한 시퀀스 생성
def create_sequences(data, target_col, seq_length=168, pred_length=24):
    """
    시계열 예측을 위한 시퀀스 생성

    인자:
        data: 특징을 가진 DataFrame
        target_col: 대상 컴럼 이름
        seq_length: 입력 시퀀스 길이 (168 = 1주일)
        pred_length: 예측 길이 (24 = 1일 앞)
    """
    # timestamp 컴럼이 있으면 제거, 없으면 target_col만 제거
    cols_to_drop = [col for col in ['timestamp', target_col] if col in data.columns]
    features = data.drop(columns=cols_to_drop).values
    target = data[target_col].values
    
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length:i+seq_length+pred_length])
    
    return np.array(X), np.array(y)

# 정책 인식 LSTM 모델
class PolicyAwareLSTM(keras.Model):
    """
    전력 수요 예측을 위한 정책 인식 구성 요소가 포함된 LSTM 모델
    """
    def __init__(self, n_features, lstm_units=[64, 32], pred_length=24):
        super(PolicyAwareLSTM, self).__init__()
        
        # LSTM 층
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_seq = (i < len(lstm_units) - 1)
            self.lstm_layers.append(
                keras.layers.LSTM(units, return_sequences=return_seq, 
                                 dropout=0.2, recurrent_dropout=0.2)
            )
        
        # 주의 메커니즘
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=lstm_units[-1] // 4
        )
        
        # 밀집층
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dropout = keras.layers.Dropout(0.3)
        self.dense2 = keras.layers.Dense(pred_length)
    
    def call(self, inputs, training=None):
        x = inputs
        
        # LSTM 층 통과
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        
        # 자기 주의
        attended = self.attention(
            tf.expand_dims(x, 1),
            tf.expand_dims(x, 1),
            tf.expand_dims(x, 1)
        )
        x = tf.squeeze(attended, 1)
        
        # 밀집층
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.dense2(x)
        
        return output

# 모델 구축 및 컴파일
def build_lstm_model(input_shape, pred_length=24):
    """
    수요 예측을 위한 LSTM 모델 구축 및 컴파일
    """
    model = PolicyAwareLSTM(
        n_features=input_shape[-1],
        lstm_units=[128, 64, 32],
        pred_length=pred_length
    )
    
    # 커스텀 학습률 스케줄
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    # 커스텀 손실로 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mse',
        metrics=['mae', keras.metrics.MeanAbsolutePercentageError()]
    )
    
    return model

# 정책 단계 모니터링을 위한 커스텀 콜백
class PolicyPhaseCallback(keras.callbacks.Callback):
    """
    정책 단계에 걸쳐 성능을 모니터링하는 커스텀 콜백
    """
    def __init__(self, validation_data, policy_phases):
        super().__init__()
        self.validation_data = validation_data
        self.policy_phases = policy_phases
        self.phase_metrics = {phase: [] for phase in range(5)}
    
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val, verbose=0)
        
        for phase in range(5):
            phase_mask = (self.policy_phases == phase)
            if np.any(phase_mask):
                phase_mae = np.mean(np.abs(y_val[phase_mask] - predictions[phase_mask]))
                self.phase_metrics[phase].append(phase_mae)
                
        if epoch % 10 == 0:
            print(f"\nPolicy Phase Performance (MAE):")
            for phase, metrics in self.phase_metrics.items():
                if metrics:
                    print(f"  Phase {phase}: {metrics[-1]:.2f}")

# 시각화 함수
def visualize_training_history(history):
    """
    훈련 이력 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Model Loss During Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE 그래프
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error During Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/chapter3_training_history.png', dpi=150)
    plt.show()

def visualize_predictions(model, X_test, y_test, n_samples=3):
    """
    모델 예측값 vs 실제값 시각화
    """
    predictions = model.predict(X_test)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(X_test))
        
        axes[i].plot(y_test[idx], label='Actual', linewidth=2)
        axes[i].plot(predictions[idx], label='Predicted', linewidth=2, linestyle='--')
        axes[i].set_xlabel('Hours Ahead')
        axes[i].set_ylabel('Electricity Demand (Normalized)')
        axes[i].set_title(f'24-Hour Ahead Forecast - Sample {i+1}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/chapter3_predictions.png', dpi=150)
    plt.show()

def analyze_policy_impact(model, demand_df):
    """
    예측에 대한 정책 개입의 영향 분석
    """
    # 정책 개입 기간 식별
    policy_periods = demand_df.groupby('policy_phase')['demand_mw'].agg(['mean', 'std'])
    
    print("\n📊 Policy Phase Analysis:")
    print(policy_periods)
    
    # 정책 단계별 예측 정확도 계산
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 정책 단계별 수요 분포
    demand_df.boxplot(column='demand_mw', by='policy_phase', ax=axes[0])
    axes[0].set_xlabel('Policy Phase')
    axes[0].set_ylabel('Demand (MW)')
    axes[0].set_title('Demand Distribution by Policy Phase')
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # 시간에 따른 재생 에너지 발전
    axes[1].plot(demand_df.index[:1000], demand_df['solar_generation_mw'].iloc[:1000], 
                label='Solar', alpha=0.7)
    axes[1].plot(demand_df.index[:1000], demand_df['wind_generation_mw'].iloc[:1000], 
                label='Wind', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Generation (MW)')
    axes[1].set_title('Renewable Generation Pattern')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('output/chapter3_policy_analysis.png', dpi=150)
    plt.show()

# 메인 실행
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 시작: 딥러닝 기반 정책 시계열 예측 분석")
    print("="*60)

    # 데이터 디렉토리 설정
    DATA_DIR = 'data'
    OUTPUT_DIR = 'output'

    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 데이터 로드 (필요시 자동 생성)
    print("\n📂 데이터 로드 중...")
    demand_df, policy_df = load_and_prepare_data(data_dir=DATA_DIR, generate_if_missing=True)

    # 모델링을 위한 특징 선택
    print("\n🎯 특징 선택 및 전처리 중...")

    # hour_sin, hour_cos 특징이 있는지 확인하고 없으면 생성
    if 'hour_sin' not in demand_df.columns:
        demand_df['hour'] = demand_df['timestamp'].dt.hour
        demand_df['hour_sin'] = np.sin(2 * np.pi * demand_df['hour'] / 24)
        demand_df['hour_cos'] = np.cos(2 * np.pi * demand_df['hour'] / 24)

    if 'day_sin' not in demand_df.columns:
        demand_df['day'] = demand_df['timestamp'].dt.dayofyear
        demand_df['day_sin'] = np.sin(2 * np.pi * demand_df['day'] / 365)
        demand_df['day_cos'] = np.cos(2 * np.pi * demand_df['day'] / 365)

    # 특징 컬럼 선택
    feature_cols = [
        'demand_mw', 'solar_generation_mw', 'wind_generation_mw', 'temperature',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_holiday', 'demand_lag_24', 'demand_lag_48',
        'demand_lag_168', 'demand_ma_24', 'demand_std_24'
    ]

    # 사용 가능한 특징만 필터링
    available_features = [col for col in feature_cols if col in demand_df.columns]

    # 특징 정규화
    scaler = MinMaxScaler()
    demand_df[available_features] = scaler.fit_transform(demand_df[available_features])

    print(f"✅ {len(available_features)}개 특징 사용: {', '.join(available_features[:5])}...")

    # 시퀀스 생성
    print("\n📦 시퀀스 생성 중...")

    # 시퀀스 생성을 위해 demand_mw를 특징에 포함
    sequence_cols = available_features if 'demand_mw' in available_features else available_features + ['demand_mw']
    X, y = create_sequences(demand_df[sequence_cols], 'demand_mw')
    print(f"✅ 시퀀스 생성 완료: X shape {X.shape}, y shape {y.shape}")

    # 데이터 분할: 70% 훈련, 15% 검증, 15% 테스트
    print("\n🔀 데이터 분할 중...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, shuffle=False)

    print(f"✅ 데이터 분할 완료:")
    print(f"   - 훈련: {X_train.shape[0]} 샘플")
    print(f"   - 검증: {X_val.shape[0]} 샘플")
    print(f"   - 테스트: {X_test.shape[0]} 샘플")

    # 모델 초기화
    print("\n🏗️ LSTM 모델 구축 중...")
    lstm_model = build_lstm_model(X_train.shape[1:])

    # 한 번 호출하여 모델 구축
    _ = lstm_model(X_train[:1])

    print("✅ 모델 아키텍처:")
    print(f"   총 파라미터 수: {sum([tf.size(w).numpy() for w in lstm_model.trainable_weights]):,}")

    # 훈련 설정
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'lstm_best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # 모델 훈련 (데모를 위해 에포크 감소)
    print("\n🚀 모델 훈련 시작...")
    print("-" * 40)

    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # 데모를 위해 감소
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 테스트 세트에서 평가
    print("\n📊 모델 평가 중...")
    test_loss, test_mae, test_mape = lstm_model.evaluate(X_test, y_test)
    print(f"\n📊 테스트 성능:")
    print(f"   손실 (MSE): {test_loss:.4f}")
    print(f"   MAE: {test_mae:.2f}")
    print(f"   MAPE: {test_mape:.2f}%")

    # 시각화 실행
    print("\n📈 분석 결과 시각화 중...")

    # 학습 이력 시각화
    visualize_training_history(history)

    # 예측 결과 시각화
    visualize_predictions(lstm_model, X_test, y_test, n_samples=2)

    # 정책 영향 분석
    analyze_policy_impact(lstm_model, demand_df)

    # 결과 요약
    print("\n" + "="*60)
    print("🎆 분석 완료!")
    print("="*60)
    print(f"\n📁 결과 저장 위치:")
    print(f"   - 데이터: {DATA_DIR}/")
    print(f"   - 모델: {OUTPUT_DIR}/lstm_best_model.keras")
    print(f"   - 그래프: {OUTPUT_DIR}/")
    print("\n✅ 딥러닝 기반 정책 시계열 예측 분석이 성공적으로 완료되었습니다!")
    print("="*60)