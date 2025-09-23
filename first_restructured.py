"""
제3장: 딥러닝 기초와 정책 시계열 예측
교육용 시각화와 실제 분석을 분리한 구조화된 버전
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
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 필요한 디렉토리 생성
os.makedirs('output', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# =====================================================
# 사용자 모드 선택
# =====================================================

def select_mode():
    """실행 모드 선택"""
    print("\n" + "="*60)
    print("딥러닝 정책 시계열 예측 시스템")
    print("="*60)
    print("\n실행 모드를 선택하세요:")
    print("1. 교육 모드 (개념 설명 및 시각화)")
    print("2. 실전 분석 모드 (데이터 분석)")
    print("3. 전체 실행 (교육 + 분석)")

    while True:
        try:
            choice = input("\n선택 (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("올바른 선택지를 입력하세요 (1, 2, 또는 3)")
        except:
            print("잘못된 입력입니다. 다시 시도하세요.")

# =====================================================
# Part 1: 교육용 시각화 함수들
# =====================================================

def demonstrate_neural_networks():
    """신경망 개념을 시각화하여 설명"""
    print("\n" + "="*50)
    print("신경망 구조 시각화")
    print("="*50)

    # 간단한 신경망 구조 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 퍼셉트론 시각화
    ax = axes[0]
    ax.set_title('단일 퍼셉트론', fontsize=12)

    # 입력 노드
    for i in range(3):
        circle = plt.Circle((0.2, 0.3 + i*0.2), 0.05, color='lightblue', ec='black')
        ax.add_patch(circle)
        ax.text(0.05, 0.3 + i*0.2, f'x{i+1}', fontsize=10)
        ax.arrow(0.25, 0.3 + i*0.2, 0.3, 0.1 - i*0.05, head_width=0.02, head_length=0.02, fc='gray')

    # 출력 노드
    circle = plt.Circle((0.7, 0.5), 0.05, color='lightgreen', ec='black')
    ax.add_patch(circle)
    ax.text(0.85, 0.5, 'y', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 2. 다층 퍼셉트론 시각화
    ax = axes[1]
    ax.set_title('다층 퍼셉트론 (MLP)', fontsize=12)

    layers = [3, 4, 2, 1]
    layer_positions = [0.2, 0.4, 0.6, 0.8]

    for l_idx, (layer_size, x_pos) in enumerate(zip(layers, layer_positions)):
        for n_idx in range(layer_size):
            y_pos = 0.5 + (n_idx - layer_size/2) * 0.15

            if l_idx == 0:
                color = 'lightblue'
            elif l_idx == len(layers) - 1:
                color = 'lightgreen'
            else:
                color = 'lightyellow'

            circle = plt.Circle((x_pos, y_pos), 0.03, color=color, ec='black')
            ax.add_patch(circle)

            # 연결선 그리기
            if l_idx < len(layers) - 1:
                next_layer_size = layers[l_idx + 1]
                next_x_pos = layer_positions[l_idx + 1]
                for next_idx in range(next_layer_size):
                    next_y_pos = 0.5 + (next_idx - next_layer_size/2) * 0.15
                    ax.plot([x_pos, next_x_pos], [y_pos, next_y_pos],
                           'gray', alpha=0.3, linewidth=0.5)

    ax.text(0.2, 0.05, '입력층', fontsize=10, ha='center')
    ax.text(0.5, 0.05, '은닉층', fontsize=10, ha='center')
    ax.text(0.8, 0.05, '출력층', fontsize=10, ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 3. 활성화 함수 시각화
    ax = axes[2]
    ax.set_title('주요 활성화 함수', fontsize=12)

    x = np.linspace(-3, 3, 100)

    # ReLU
    relu = np.maximum(0, x)
    ax.plot(x, relu, label='ReLU', linewidth=2)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    ax.plot(x, sigmoid, label='Sigmoid', linewidth=2)

    # Tanh
    tanh = np.tanh(x)
    ax.plot(x, tanh, label='Tanh', linewidth=2)

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('입력값')
    ax.set_ylabel('출력값')

    plt.tight_layout()
    plt.savefig('visualizations/neural_networks_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 신경망 구조 시각화 완료 (visualizations/neural_networks_demo.png)")

def demonstrate_time_series_concepts():
    """시계열 분석 개념을 시각화하여 설명"""
    print("\n" + "="*50)
    print("시계열 분석 개념 시각화")
    print("="*50)

    # 샘플 시계열 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')

    # 트렌드 + 계절성 + 노이즈
    trend = np.linspace(100, 150, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
    noise = np.random.normal(0, 5, 365)
    ts = trend + seasonal + noise

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 원본 시계열
    ax = axes[0, 0]
    ax.plot(dates, ts, color='blue', alpha=0.7)
    ax.set_title('시계열 데이터 예시', fontsize=12)
    ax.set_xlabel('날짜')
    ax.set_ylabel('값')
    ax.grid(True, alpha=0.3)

    # 2. 시계열 분해
    ax = axes[0, 1]
    ax.plot(dates, trend, label='트렌드', linewidth=2, color='red')
    ax.plot(dates, seasonal + 125, label='계절성', linewidth=2, color='green')
    ax.plot(dates, noise + 100, label='노이즈', linewidth=1, color='gray', alpha=0.5)
    ax.set_title('시계열 구성 요소', fontsize=12)
    ax.set_xlabel('날짜')
    ax.set_ylabel('값')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 자기상관 함수 (ACF)
    ax = axes[1, 0]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(ts, lags=50, ax=ax)
    ax.set_title('자기상관 함수 (ACF)', fontsize=12)

    # 4. 정책 개입 효과 시각화
    ax = axes[1, 1]

    # 정책 개입 시뮬레이션
    policy_start = 200
    ts_with_policy = ts.copy()
    ts_with_policy[policy_start:] += 20  # 정책 효과

    ax.plot(dates, ts, label='정책 개입 전', color='blue', alpha=0.7)
    ax.plot(dates, ts_with_policy, label='정책 개입 후', color='red', alpha=0.7)
    ax.axvline(x=dates[policy_start], color='green', linestyle='--', label='정책 시행일')
    ax.set_title('정책 개입 효과', fontsize=12)
    ax.set_xlabel('날짜')
    ax.set_ylabel('값')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/time_series_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 시계열 개념 시각화 완료 (visualizations/time_series_demo.png)")

def demonstrate_rnn_concepts():
    """RNN 개념을 시각화하여 설명"""
    print("\n" + "="*50)
    print("RNN 구조 시각화")
    print("="*50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 기본 RNN 구조
    ax = axes[0]
    ax.set_title('기본 RNN 구조', fontsize=12)

    # RNN 셀 그리기
    for t in range(3):
        x_pos = 0.3 + t * 0.2

        # 입력
        ax.arrow(x_pos, 0.2, 0, 0.15, head_width=0.02, head_length=0.02, fc='blue')
        ax.text(x_pos, 0.15, f'x{t}', fontsize=10, ha='center')

        # RNN 셀
        rect = plt.Rectangle((x_pos-0.05, 0.4), 0.1, 0.2,
                            facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos, 0.5, 'RNN', fontsize=10, ha='center')

        # 출력
        ax.arrow(x_pos, 0.6, 0, 0.15, head_width=0.02, head_length=0.02, fc='red')
        ax.text(x_pos, 0.8, f'h{t}', fontsize=10, ha='center')

        # 은닉 상태 연결
        if t < 2:
            ax.arrow(x_pos+0.05, 0.5, 0.1, 0, head_width=0.02, head_length=0.02, fc='gray')

    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 2. Vanishing Gradient 문제
    ax = axes[1]
    ax.set_title('Gradient Vanishing 문제', fontsize=12)

    timesteps = np.arange(1, 11)
    gradient_flow = 0.5 ** timesteps  # 지수적 감소

    ax.bar(timesteps, gradient_flow, color='red', alpha=0.7)
    ax.set_xlabel('시간 단계')
    ax.set_ylabel('Gradient 크기')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. 시퀀스 길이별 성능
    ax = axes[2]
    ax.set_title('시퀀스 길이에 따른 모델 성능', fontsize=12)

    seq_lengths = np.array([10, 20, 30, 50, 100, 200])
    rnn_performance = 0.9 * np.exp(-seq_lengths/50)
    lstm_performance = 0.9 - 0.1 * seq_lengths/200

    ax.plot(seq_lengths, rnn_performance, 'o-', label='기본 RNN', linewidth=2)
    ax.plot(seq_lengths, lstm_performance, 's-', label='LSTM', linewidth=2)
    ax.set_xlabel('시퀀스 길이')
    ax.set_ylabel('정확도')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/rnn_concepts_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ RNN 개념 시각화 완료 (visualizations/rnn_concepts_demo.png)")

def demonstrate_lstm_gru():
    """LSTM과 GRU 구조를 시각화하여 설명"""
    print("\n" + "="*50)
    print("LSTM/GRU 구조 시각화")
    print("="*50)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. LSTM 게이트 동작
    ax = axes[0, 0]
    ax.set_title('LSTM 게이트 메커니즘', fontsize=12)

    gates = ['Forget Gate', 'Input Gate', 'Output Gate']
    values = [0.7, 0.9, 0.6]
    colors = ['red', 'blue', 'green']

    bars = ax.bar(gates, values, color=colors, alpha=0.7)
    ax.set_ylabel('게이트 값')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom')

    # 2. GRU vs LSTM 파라미터 수
    ax = axes[0, 1]
    ax.set_title('모델 복잡도 비교', fontsize=12)

    models = ['RNN', 'GRU', 'LSTM']
    params = [100, 300, 400]  # 상대적 파라미터 수

    bars = ax.bar(models, params, color=['gray', 'orange', 'purple'], alpha=0.7)
    ax.set_ylabel('파라미터 수 (상대값)')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 장기 의존성 학습 능력
    ax = axes[1, 0]
    ax.set_title('장기 의존성 학습 능력', fontsize=12)

    distance = np.arange(1, 101)
    rnn_ability = np.exp(-distance/10)
    lstm_ability = np.exp(-distance/50)
    gru_ability = np.exp(-distance/40)

    ax.plot(distance, rnn_ability, label='RNN', linewidth=2)
    ax.plot(distance, lstm_ability, label='LSTM', linewidth=2)
    ax.plot(distance, gru_ability, label='GRU', linewidth=2)
    ax.set_xlabel('시간 간격')
    ax.set_ylabel('정보 보존 능력')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 학습 속도 비교
    ax = axes[1, 1]
    ax.set_title('학습 수렴 속도', fontsize=12)

    epochs = np.arange(1, 51)
    rnn_loss = 0.5 * np.exp(-epochs/10) + 0.15
    lstm_loss = 0.5 * np.exp(-epochs/15) + 0.05
    gru_loss = 0.5 * np.exp(-epochs/12) + 0.08

    ax.plot(epochs, rnn_loss, label='RNN', linewidth=2)
    ax.plot(epochs, lstm_loss, label='LSTM', linewidth=2)
    ax.plot(epochs, gru_loss, label='GRU', linewidth=2)
    ax.set_xlabel('에폭')
    ax.set_ylabel('손실값')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/lstm_gru_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ LSTM/GRU 구조 시각화 완료 (visualizations/lstm_gru_demo.png)")

# =====================================================
# Part 2: 실제 데이터 처리 및 분석 함수들
# =====================================================

def generate_and_save_data(output_dir='data'):
    """합성 데이터를 생성하고 CSV 파일로 저장"""
    print("\n" + "="*50)
    print("데이터 생성 및 저장")
    print("="*50)

    np.random.seed(42)

    # 1. 전력 수요 데이터 생성
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='H')
    n_hours = len(dates)

    # 기본 패턴
    hourly_pattern = np.array([0.7, 0.65, 0.6, 0.58, 0.57, 0.58,  # 0-5시
                               0.65, 0.75, 0.85, 0.9, 0.92, 0.94,   # 6-11시
                               0.93, 0.92, 0.91, 0.9, 0.88, 0.87,   # 12-17시
                               0.9, 0.95, 0.97, 0.85, 0.8, 0.75])   # 18-23시

    # 계절 패턴
    day_of_year = dates.dayofyear
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)

    # 전력 수요 생성
    base_demand = 50000  # MW
    demand = []
    for i, date in enumerate(dates):
        hour = date.hour
        daily_factor = hourly_pattern[hour]
        season_factor = seasonal_factor[i % len(seasonal_factor)]
        noise = np.random.normal(0, 0.05)
        demand_value = base_demand * daily_factor * season_factor * (1 + noise)
        demand.append(demand_value)

    demand_df = pd.DataFrame({
        'timestamp': dates,
        'demand_mw': demand,
        'temperature': 15 + 10 * np.sin(2 * np.pi * day_of_year[:n_hours] / 365) + np.random.normal(0, 3, n_hours),
        'humidity': 60 + 20 * np.sin(2 * np.pi * day_of_year[:n_hours] / 365 + np.pi/4) + np.random.normal(0, 5, n_hours),
        'is_weekend': dates.weekday.isin([5, 6]).astype(int)
    })

    # 2. 재생에너지 정책 데이터 생성
    policy_changes = pd.date_range('2022-01-01', '2023-12-31', freq='M')
    policy_levels = np.cumsum(np.random.uniform(0, 2, len(policy_changes)))

    policy_df = pd.DataFrame({
        'timestamp': policy_changes,
        'renewable_target': 20 + policy_levels,
        'subsidy_rate': 0.1 + 0.01 * policy_levels,
        'carbon_tax': 10 + 2 * policy_levels
    })

    # 3. 전력 시장 데이터 생성
    market_df = pd.DataFrame({
        'timestamp': dates,
        'smp': 80 + 20 * np.sin(2 * np.pi * day_of_year[:n_hours] / 365) + np.random.normal(0, 10, n_hours),
        'rec_price': 50 + 10 * np.sin(2 * np.pi * day_of_year[:n_hours] / 365 + np.pi/3) + np.random.normal(0, 5, n_hours),
        'lng_price': 10 + 2 * np.sin(2 * np.pi * day_of_year[:n_hours] / 365 - np.pi/4) + np.random.normal(0, 1, n_hours)
    })

    # 데이터 저장
    demand_df.to_csv(f'{output_dir}/energy_demand.csv', index=False)
    policy_df.to_csv(f'{output_dir}/renewable_policy.csv', index=False)
    market_df.to_csv(f'{output_dir}/electricity_market.csv', index=False)

    print(f"✅ 전력 수요 데이터 저장: {output_dir}/energy_demand.csv")
    print(f"✅ 재생에너지 정책 데이터 저장: {output_dir}/renewable_policy.csv")
    print(f"✅ 전력 시장 데이터 저장: {output_dir}/electricity_market.csv")

    return demand_df, policy_df, market_df

def load_and_prepare_data(data_dir='data', generate_if_missing=True):
    """저장된 데이터를 로드하고 모델링을 위해 준비"""

    # 파일 확인
    required_files = ['energy_demand.csv', 'renewable_policy.csv', 'electricity_market.csv']
    files_exist = all(os.path.exists(f'{data_dir}/{f}') for f in required_files)

    if not files_exist:
        if generate_if_missing:
            print("⚠️ 데이터 파일이 없습니다. 새로 생성합니다...")
            generate_and_save_data(data_dir)
        else:
            raise FileNotFoundError("필요한 데이터 파일이 없습니다.")

    # 데이터 로드
    print("\n📊 데이터 로딩 중...")
    demand_df = pd.read_csv(f'{data_dir}/energy_demand.csv', parse_dates=['timestamp'])
    policy_df = pd.read_csv(f'{data_dir}/renewable_policy.csv', parse_dates=['timestamp'])
    market_df = pd.read_csv(f'{data_dir}/electricity_market.csv', parse_dates=['timestamp'])

    # 데이터 병합
    merged_df = demand_df.merge(market_df, on='timestamp', how='left')
    merged_df = pd.merge_asof(merged_df.sort_values('timestamp'),
                              policy_df.sort_values('timestamp'),
                              on='timestamp',
                              direction='backward')
    merged_df = merged_df.fillna(method='ffill')

    print(f"✅ 데이터 로드 완료: {len(merged_df)} 레코드")

    return merged_df

def create_sequences(data, sequence_length=24, target_col='demand_mw'):
    """시계열 예측을 위한 시퀀스 생성"""

    # timestamp 열이 있으면 제외
    cols_to_drop = [col for col in ['timestamp', target_col] if col in data.columns]
    features = data.drop(columns=cols_to_drop).values
    targets = data[target_col].values

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])

    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """LSTM 모델 구축"""
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    """GRU 모델 구축"""
    model = keras.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_models(data):
    """모델 학습 및 평가"""
    print("\n" + "="*50)
    print("모델 학습 및 평가")
    print("="*50)

    # 데이터 전처리
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # 시퀀스 생성
    X, y = create_sequences(data, sequence_length=24)

    # 학습/검증/테스트 분할
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"학습 데이터: {X_train.shape}")
    print(f"검증 데이터: {X_val.shape}")
    print(f"테스트 데이터: {X_test.shape}")

    # LSTM 모델 학습
    print("\n🔄 LSTM 모델 학습 중...")
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # GRU 모델 학습
    print("\n🔄 GRU 모델 학습 중...")
    gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))

    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # 모델 평가
    print("\n📊 모델 성능 평가")
    lstm_test_loss, lstm_test_mae = lstm_model.evaluate(X_test, y_test, verbose=0)
    gru_test_loss, gru_test_mae = gru_model.evaluate(X_test, y_test, verbose=0)

    print(f"LSTM - Test Loss: {lstm_test_loss:.4f}, Test MAE: {lstm_test_mae:.4f}")
    print(f"GRU  - Test Loss: {gru_test_loss:.4f}, Test MAE: {gru_test_mae:.4f}")

    # 예측 시각화
    visualize_predictions(lstm_model, gru_model, X_test, y_test, scaler)

    return lstm_model, gru_model, lstm_history, gru_history

def visualize_predictions(lstm_model, gru_model, X_test, y_test, scaler):
    """예측 결과 시각화"""

    # 예측
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 샘플 구간 선택 (처음 200개)
    n_show = min(200, len(y_test))

    # LSTM 예측
    ax = axes[0]
    ax.plot(y_test[:n_show], label='실제값', alpha=0.7)
    ax.plot(lstm_pred[:n_show], label='LSTM 예측', alpha=0.7)
    ax.set_title('LSTM 모델 예측 결과')
    ax.set_xlabel('시간')
    ax.set_ylabel('전력 수요 (정규화)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GRU 예측
    ax = axes[1]
    ax.plot(y_test[:n_show], label='실제값', alpha=0.7)
    ax.plot(gru_pred[:n_show], label='GRU 예측', alpha=0.7)
    ax.set_title('GRU 모델 예측 결과')
    ax.set_xlabel('시간')
    ax.set_ylabel('전력 수요 (정규화)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/model_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 예측 결과 시각화 완료 (output/model_predictions.png)")

def analyze_policy_impact(data, model):
    """정책 영향 분석"""
    print("\n" + "="*50)
    print("정책 영향 분석")
    print("="*50)

    # 정책 변화 시점 찾기
    policy_cols = ['renewable_target', 'subsidy_rate', 'carbon_tax']

    fig, axes = plt.subplots(len(policy_cols), 1, figsize=(15, 10))

    for idx, col in enumerate(policy_cols):
        if col in data.columns:
            ax = axes[idx]

            # 정책 변수와 전력 수요의 관계
            ax2 = ax.twinx()

            ax.plot(data.index[:1000], data[col].iloc[:1000],
                   color='blue', alpha=0.7, label=col)
            ax2.plot(data.index[:1000], data['demand_mw'].iloc[:1000],
                    color='red', alpha=0.5, label='전력 수요')

            ax.set_xlabel('시간')
            ax.set_ylabel(col, color='blue')
            ax2.set_ylabel('전력 수요', color='red')
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_title(f'{col}와 전력 수요의 관계')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/policy_impact_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ 정책 영향 분석 완료 (output/policy_impact_analysis.png)")

# =====================================================
# 메인 실행 부분
# =====================================================

def main():
    """메인 실행 함수"""

    # 모드 선택
    mode = select_mode()

    # 교육 모드 실행
    if mode in [1, 3]:
        print("\n" + "="*60)
        print("교육 모드 실행")
        print("="*60)

        demonstrate_neural_networks()
        demonstrate_time_series_concepts()
        demonstrate_rnn_concepts()
        demonstrate_lstm_gru()

        print("\n✅ 교육 모드 완료!")

    # 실전 분석 모드 실행
    if mode in [2, 3]:
        print("\n" + "="*60)
        print("실전 분석 모드 실행")
        print("="*60)

        # 데이터 준비
        data = load_and_prepare_data()

        # 모델 학습 및 평가
        lstm_model, gru_model, lstm_hist, gru_hist = train_and_evaluate_models(data)

        # 정책 영향 분석
        analyze_policy_impact(data, lstm_model)

        print("\n✅ 실전 분석 모드 완료!")

    print("\n" + "="*60)
    print("프로그램 종료")
    print("="*60)

if __name__ == "__main__":
    main()