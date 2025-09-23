"""
제3장: 딥러닝 기초와 정책 시계열 예측
완전 자동 실행 버전 - 사용자 입력 없이 모든 분석 수행
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
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

print("\n" + "="*60)
print("딥러닝 정책 시계열 예측 시스템 - 자동 실행")
print("="*60)

# =====================================================
# Part 1: 교육용 시각화 함수들
# =====================================================

def demonstrate_neural_networks():
    """신경망 개념을 시각화하여 설명"""
    print("\n[1/5] 신경망 구조 시각화 생성 중...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 퍼셉트론 시각화
    ax = axes[0]
    ax.set_title('단일 퍼셉트론', fontsize=12)

    # 입력 노드
    for i in range(3):
        circle = plt.Circle((0.2, 0.3 + i*0.2), 0.05, color='lightblue', ec='black')
        ax.add_patch(circle)
        ax.text(0.05, 0.3 + i*0.2, f'x{i+1}', fontsize=10)

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
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    ax.plot(x, relu, label='ReLU', linewidth=2)
    ax.plot(x, sigmoid, label='Sigmoid', linewidth=2)
    ax.plot(x, tanh, label='Tanh', linewidth=2)

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('입력값')
    ax.set_ylabel('출력값')

    plt.tight_layout()
    plt.savefig('visualizations/neural_networks_demo.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ 신경망 구조 시각화 저장 완료")

def demonstrate_time_series_concepts():
    """시계열 분석 개념을 시각화하여 설명"""
    print("\n[2/5] 시계열 분석 개념 시각화 생성 중...")

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

    # 3. 자기상관 함수 (간단한 버전)
    ax = axes[1, 0]
    lags = range(50)
    acf_values = [np.corrcoef(ts[:-lag-1], ts[lag+1:])[0,1] if lag > 0 else 1.0 for lag in lags]
    ax.bar(lags, acf_values, color='blue', alpha=0.7)
    ax.set_title('자기상관 함수 (ACF)', fontsize=12)
    ax.set_xlabel('시차')
    ax.set_ylabel('상관계수')
    ax.grid(True, alpha=0.3)

    # 4. 정책 개입 효과 시각화
    ax = axes[1, 1]
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
    plt.close()

    print("   ✅ 시계열 개념 시각화 저장 완료")

def demonstrate_lstm_gru():
    """LSTM과 GRU 성능 비교를 시각화하여 설명"""
    print("\n[3/5] LSTM/GRU 성능 비교 시각화 생성 중...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 장기 의존성 학습 능력
    ax = axes[0]
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

    # 2. 학습 속도 비교
    ax = axes[1]
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
    plt.savefig('visualizations/lstm_gru_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ LSTM/GRU 비교 시각화 저장 완료")

# =====================================================
# Part 2: 실제 데이터 처리 및 분석 함수들
# =====================================================

def generate_and_save_data(output_dir='data'):
    """합성 데이터를 생성하고 CSV 파일로 저장"""
    print("\n[4/5] 데이터 생성 및 저장 중...")

    np.random.seed(42)

    # 1. 전력 수요 데이터 생성 (1개월 분량만 - 빠른 실행을 위해)
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='H')
    n_hours = len(dates)

    # 기본 패턴
    hourly_pattern = np.array([0.7, 0.65, 0.6, 0.58, 0.57, 0.58,  # 0-5시
                               0.65, 0.75, 0.85, 0.9, 0.92, 0.94,   # 6-11시
                               0.93, 0.92, 0.91, 0.9, 0.88, 0.87,   # 12-17시
                               0.9, 0.95, 0.97, 0.85, 0.8, 0.75])   # 18-23시

    # 전력 수요 생성
    base_demand = 50000  # MW
    demand = []
    for i, date in enumerate(dates):
        hour = date.hour
        daily_factor = hourly_pattern[hour]
        noise = np.random.normal(0, 0.05)
        demand_value = base_demand * daily_factor * (1 + noise)
        demand.append(demand_value)

    demand_df = pd.DataFrame({
        'timestamp': dates,
        'demand_mw': demand,
        'temperature': 5 + np.random.normal(0, 3, n_hours),  # 겨울 온도
        'humidity': 60 + np.random.normal(0, 5, n_hours),
        'is_weekend': dates.weekday.isin([5, 6]).astype(int)
    })

    # 2. 재생에너지 정책 데이터 생성
    policy_changes = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    policy_levels = np.cumsum(np.random.uniform(0, 0.5, len(policy_changes)))

    policy_df = pd.DataFrame({
        'timestamp': policy_changes,
        'renewable_target': 20 + policy_levels,
        'subsidy_rate': 0.1 + 0.001 * policy_levels,
        'carbon_tax': 10 + 0.5 * policy_levels
    })

    # 3. 전력 시장 데이터 생성
    market_df = pd.DataFrame({
        'timestamp': dates,
        'smp': 100 + np.random.normal(0, 10, n_hours),
        'rec_price': 50 + np.random.normal(0, 5, n_hours),
        'lng_price': 12 + np.random.normal(0, 1, n_hours)
    })

    # 데이터 저장
    demand_df.to_csv(f'{output_dir}/energy_demand.csv', index=False)
    policy_df.to_csv(f'{output_dir}/renewable_policy.csv', index=False)
    market_df.to_csv(f'{output_dir}/electricity_market.csv', index=False)

    print(f"   ✅ 전력 수요 데이터 저장: {output_dir}/energy_demand.csv")
    print(f"   ✅ 재생에너지 정책 데이터 저장: {output_dir}/renewable_policy.csv")
    print(f"   ✅ 전력 시장 데이터 저장: {output_dir}/electricity_market.csv")

    return demand_df, policy_df, market_df

def quick_analysis_and_visualization(demand_df, policy_df, market_df):
    """빠른 분석 및 시각화"""
    print("\n[5/5] 데이터 분석 및 시각화 생성 중...")

    # 데이터 병합
    merged_df = demand_df.merge(market_df, on='timestamp', how='left')

    # 간단한 통계
    print("\n📊 데이터 요약 통계:")
    print(f"   - 평균 전력 수요: {demand_df['demand_mw'].mean():.0f} MW")
    print(f"   - 최대 전력 수요: {demand_df['demand_mw'].max():.0f} MW")
    print(f"   - 최소 전력 수요: {demand_df['demand_mw'].min():.0f} MW")
    print(f"   - 평균 SMP: {market_df['smp'].mean():.1f} 원/kWh")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 일별 전력 수요 패턴
    ax = axes[0, 0]
    daily_demand = demand_df.groupby(demand_df['timestamp'].dt.date)['demand_mw'].mean()
    ax.plot(daily_demand.index, daily_demand.values, linewidth=2, color='blue')
    ax.set_title('일별 평균 전력 수요', fontsize=12)
    ax.set_xlabel('날짜')
    ax.set_ylabel('평균 수요 (MW)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 2. 시간대별 평균 수요
    ax = axes[0, 1]
    hourly_demand = demand_df.groupby(demand_df['timestamp'].dt.hour)['demand_mw'].mean()
    ax.plot(hourly_demand.index, hourly_demand.values, linewidth=2, color='red', marker='o')
    ax.set_title('시간대별 평균 전력 수요', fontsize=12)
    ax.set_xlabel('시간')
    ax.set_ylabel('평균 수요 (MW)')
    ax.grid(True, alpha=0.3)

    # 3. 수요와 온도의 관계
    ax = axes[1, 0]
    ax.scatter(demand_df['temperature'], demand_df['demand_mw'], alpha=0.5, s=10)
    ax.set_title('온도와 전력 수요의 관계', fontsize=12)
    ax.set_xlabel('온도 (°C)')
    ax.set_ylabel('수요 (MW)')
    ax.grid(True, alpha=0.3)

    # 4. SMP 가격 추이
    ax = axes[1, 1]
    ax.plot(market_df['timestamp'][:24*7], market_df['smp'][:24*7], linewidth=1.5, color='green')
    ax.set_title('첫 주 SMP 가격 추이', fontsize=12)
    ax.set_xlabel('시간')
    ax.set_ylabel('SMP (원/kWh)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('output/analysis_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ 분석 결과 시각화 저장 완료")

def build_and_train_simple_model(demand_df):
    """간단한 LSTM 모델 학습 (데모용)"""
    print("\n🤖 간단한 LSTM 모델 학습 중...")

    # 데이터 준비
    data = demand_df[['demand_mw', 'temperature', 'humidity', 'is_weekend']].values

    # 정규화
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 시퀀스 생성 (24시간 단위)
    sequence_length = 24
    X, y = [], []

    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length, 0])  # demand_mw만 예측

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("   ⚠️ 데이터 부족으로 모델 학습 생략")
        return

    # 학습/테스트 분할
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 간단한 LSTM 모델
    model = keras.Sequential([
        keras.layers.LSTM(32, input_shape=(sequence_length, 4)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 학습 (에폭 수를 줄여서 빠르게 실행)
    print("   모델 학습 시작...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=3,  # 빠른 실행을 위해 3 에폭만
        batch_size=32,
        verbose=0  # 출력 최소화
    )

    # 평가
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   ✅ 모델 학습 완료 - Test MAE: {test_mae:.4f}")

    # 예측 시각화
    predictions = model.predict(X_test, verbose=0)

    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:100], label='실제값', alpha=0.7)
    plt.plot(predictions[:100], label='예측값', alpha=0.7)
    plt.title('LSTM 모델 예측 결과 (처음 100개 샘플)')
    plt.xlabel('시간')
    plt.ylabel('전력 수요 (정규화)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/lstm_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   ✅ 예측 결과 시각화 저장 완료")

# =====================================================
# 메인 실행 부분
# =====================================================

def main():
    """메인 실행 함수 - 완전 자동 실행"""

    start_time = datetime.now()

    print("\n🚀 프로그램 자동 실행 시작...")
    print(f"   시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Part 1: 교육용 시각화
    print("\n" + "="*60)
    print("Part 1: 교육용 개념 시각화")
    print("="*60)

    demonstrate_neural_networks()
    demonstrate_time_series_concepts()
    demonstrate_lstm_gru()

    # Part 2: 데이터 생성 및 분석
    print("\n" + "="*60)
    print("Part 2: 데이터 생성 및 분석")
    print("="*60)

    # 데이터 생성 및 저장
    demand_df, policy_df, market_df = generate_and_save_data()

    # 데이터 분석 및 시각화
    quick_analysis_and_visualization(demand_df, policy_df, market_df)

    # 간단한 모델 학습
    build_and_train_simple_model(demand_df)

    # 완료
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    print("\n" + "="*60)
    print("✨ 모든 작업이 성공적으로 완료되었습니다!")
    print("="*60)

    print("\n📁 생성된 파일 목록:")
    print("   [교육용 시각화]")
    print("   - visualizations/neural_networks_demo.png")
    print("   - visualizations/time_series_demo.png")
    print("   - visualizations/lstm_gru_comparison.png")

    print("\n   [데이터 파일]")
    print("   - data/energy_demand.csv")
    print("   - data/renewable_policy.csv")
    print("   - data/electricity_market.csv")

    print("\n   [분석 결과]")
    print("   - output/analysis_results.png")
    print("   - output/lstm_predictions.png")

    print(f"\n⏱️ 총 실행 시간: {elapsed_time:.2f}초")
    print("\n프로그램이 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()