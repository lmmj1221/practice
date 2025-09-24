"""
딥러닝 기반 정책 시계열 예측 - 실전 분석 모듈
데이터 생성, 전처리, 모델 학습 및 평가를 수행하는 모듈
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
import joblib

warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

# 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 필요한 디렉토리 생성
os.makedirs(os.path.join('output'), exist_ok=True)
os.makedirs(os.path.join('data'), exist_ok=True)
os.makedirs(os.path.join('models'), exist_ok=True)


def load_and_prepare_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join('data')
    """저장된 데이터를 로드하고 모델링을 위해 준비"""

    # 파일 확인
    required_files = ['energy_demand.csv', 'renewable_policy.csv', 'electricity_market.csv']
    files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)

    if not files_exist:
        raise FileNotFoundError("필요한 데이터 파일이 없습니다. data/ 폴더를 확인하세요.")

    # 데이터 로드
    print("\n📊 데이터 로딩 중...")
    demand_df = pd.read_csv(os.path.join(data_dir, 'energy_demand.csv'), parse_dates=['timestamp'])
    policy_df = pd.read_csv(os.path.join(data_dir, 'renewable_policy.csv'), parse_dates=['timestamp'])
    market_df = pd.read_csv(os.path.join(data_dir, 'electricity_market.csv'), parse_dates=['timestamp'])

    # 데이터 병합
    merged_df = demand_df.merge(market_df, on='timestamp', how='left')
    merged_df = pd.merge_asof(merged_df.sort_values('timestamp'),
                              policy_df.sort_values('timestamp'),
                              on='timestamp',
                              direction='backward')
    merged_df = merged_df.ffill().fillna(0)

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
        keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def build_gru_model(input_shape):
    """GRU 모델 구축"""
    model = keras.Sequential([
        keras.layers.GRU(64, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(32, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def build_simple_rnn_model(input_shape):
    """간단한 RNN 모델 구축 (비교용)"""
    model = keras.Sequential([
        keras.layers.SimpleRNN(32, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.SimpleRNN(16),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def save_models(models, scaler, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join('models')
    """학습된 모델과 스케일러를 저장"""
    print("\n💾 모델 저장 중...")

    for name, model in models.items():
        model_path = os.path.join(save_dir, f'{name}_model.keras')
        model.save(model_path)
        print(f"✅ {name} 모델 저장: {model_path}")

    # 스케일러 저장
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ 스케일러 저장: {scaler_path}")

    print(f"\n모든 모델이 {save_dir}/ 폴더에 저장되었습니다.")

def load_models(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join('models')
    """저장된 모델과 스케일러를 로드"""
    print("\n📂 저장된 모델 로딩 중...")

    models = {}
    model_files = ['LSTM_model.keras', 'GRU_model.keras', 'RNN_model.keras']

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace('_model.keras', '')
            models[model_name] = keras.models.load_model(model_path)
            print(f"✅ {model_name} 모델 로드: {model_path}")
        else:
            print(f"⚠️ {model_file} 파일을 찾을 수 없습니다.")

    # 스케일러 로드
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ 스케일러 로드: {scaler_path}")
    else:
        print("⚠️ 스케일러 파일을 찾을 수 없습니다.")

    if not models:
        raise FileNotFoundError("로드할 수 있는 모델이 없습니다. 먼저 모델을 학습하고 저장하세요.")

    return models, scaler

def train_and_evaluate_models(data, epochs=10, save=True):
    """모델 학습 및 평가"""
    print("\n" + "="*50)
    print("모델 학습 및 평가")
    print("="*50)

    # 데이터 전처리
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # 시퀀스 생성
    X, y = create_sequences(data_scaled, sequence_length=24)

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

    models = {}
    histories = {}

    # LSTM 모델 학습
    print("\n🔄 LSTM 모델 학습 중...")
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    models['LSTM'] = lstm_model
    histories['LSTM'] = lstm_history

    # GRU 모델 학습
    print("\n🔄 GRU 모델 학습 중...")
    gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    models['GRU'] = gru_model
    histories['GRU'] = gru_history

    # 간단한 RNN 모델 학습 (옵션)
    print("\n🔄 Simple RNN 모델 학습 중...")
    rnn_model = build_simple_rnn_model((X_train.shape[1], X_train.shape[2]))
    rnn_history = rnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    models['RNN'] = rnn_model
    histories['RNN'] = rnn_history

    # 모델 평가
    print("\n📊 모델 성능 평가")
    print("-" * 50)

    results = {}
    for name, model in models.items():
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {'loss': test_loss, 'mae': test_mae}
        print(f"{name:10s} - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # 모델 저장
    if save:
        save_models(models, scaler)

    # 예측 시각화
    visualize_predictions(models, X_test, y_test, scaler)
    visualize_training_history(histories)

    return models, histories, results, scaler

def visualize_predictions(models, X_test, y_test, scaler):
    """예측 결과 시각화"""

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 5*n_models))

    if n_models == 1:
        axes = [axes]

    # 샘플 구간 선택 (처음 200개)
    n_show = min(200, len(y_test))

    for idx, (name, model) in enumerate(models.items()):
        # 예측
        pred = model.predict(X_test)

        # 시각화
        ax = axes[idx]
        ax.plot(y_test[:n_show], label='실제값', alpha=0.7, linewidth=1.5)
        ax.plot(pred[:n_show], label=f'{name} 예측', alpha=0.7, linewidth=1.5)
        ax.set_title(f'{name} 모델 예측 결과')
        ax.set_xlabel('시간')
        ax.set_ylabel('전력 수요 (정규화)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 캐픍션 추가
        mae = np.mean(np.abs(pred[:n_show].flatten() - y_test[:n_show]))
        ax.text(0.5, -0.15, f'{name} 모델의 전력 수요 예측: 24시간 이전 데이터로 다음 시간 예측\nMAE: {mae:.4f} (0에 가까울수록 정확)',
                ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'model_predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 예측 결과 시각화 완료 ({os.path.join('output', 'model_predictions.png')})")

def visualize_training_history(histories):
    """학습 과정 시각화"""

    n_models = len(histories)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, history) in enumerate(histories.items()):
        # Loss 그래프
        ax = axes[0, idx]
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Val Loss')
        ax.set_title(f'{name} - Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.2, '학습 손실: 훈련 데이터와 검증 데이터의 오차 감소 추이',
                ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

        # MAE 그래프
        ax = axes[1, idx]
        ax.plot(history.history['mae'], label='Train MAE')
        ax.plot(history.history['val_mae'], label='Val MAE')
        ax.set_title(f'{name} - MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.2, '평균 절대 오차: 예측값과 실제값의 평균적인 차이',
                ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 학습 과정 시각화 완료 ({os.path.join('output', 'training_history.png')})")

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

            # 캐플션 추가
            caption_text = {
                'renewable_target': '재생에너지 목표 증가가 전력 수요에 미치는 영향 분석',
                'subsidy_rate': '보조금 비율 변화와 전력 수요의 상관관계',
                'carbon_tax': '탄소세 정책이 전력 소비 패턴에 미치는 효과'
            }.get(col, '')

            if caption_text:
                ax.text(0.5, -0.15, caption_text,
                        ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'policy_impact_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 정책 영향 분석 완료 ({os.path.join('output', 'policy_impact_analysis.png')})")

def perform_statistical_analysis(data):
    """통계적 분석 수행"""
    print("\n" + "="*50)
    print("통계적 분석")
    print("="*50)

    # 기본 통계량
    print("\n📊 기본 통계량:")
    print("-" * 50)
    print(data.describe())

    # 상관관계 분석
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()

    # 상관관계 히트맵
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('변수 간 상관관계')

    # 히트맵 캐플션
    plt.figtext(0.5, -0.02, '상관계수: -1(완전 음의 상관) ~ 0(무관) ~ +1(완전 양의 상관)\n강한 상관관계를 보이는 변수들이 예측에 중요',
                ha='center', fontsize=9, style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 상관관계 히트맵 생성 완료 ({os.path.join('output', 'correlation_heatmap.png')})")

    # 주요 상관관계 출력
    print("\n📈 전력 수요와의 주요 상관관계:")
    print("-" * 50)
    demand_corr = correlation_matrix['demand_mw'].sort_values(ascending=False)
    for var, corr in demand_corr.items():
        if var != 'demand_mw' and abs(corr) > 0.3:
            print(f"{var:20s}: {corr:+.3f}")

def evaluate_loaded_models(models, data, scaler):
    """저장된 모델로 예측 수행"""
    print("\n" + "="*50)
    print("저장된 모델로 예측 수행")
    print("="*50)

    # 데이터 전처리
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.transform(data[numeric_columns])

    # 시퀀스 생성
    X, y = create_sequences(data_scaled, sequence_length=24)

    # 테스트 데이터 분할
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"테스트 데이터: {X_test.shape}")

    # 모델 평가
    print("\n📊 모델 성능 평가")
    print("-" * 50)

    results = {}
    for name, model in models.items():
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {'loss': test_loss, 'mae': test_mae}
        print(f"{name:10s} - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # 예측 시각화
    visualize_predictions(models, X_test, y_test, scaler)

    return results

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("딥러닝 기반 정책 시계열 예측 - 실전 분석")
    print("="*60)

    print("\n실행할 작업을 선택하세요:")
    print("1. 모델 학습 및 저장")
    print("2. 저장된 모델 로드 및 평가")
    print("3. 정책 영향 분석 (저장된 모델 사용)")
    print("4. 통계적 분석 (데이터 상관관계, 기본 통계량, 히트맵 생성)")

    while True:
        try:
            choice = input("\n선택 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                choice = int(choice)
                break
            else:
                print("올바른 선택지를 입력하세요 (1-4)")
        except:
            print("잘못된 입력입니다. 다시 시도하세요.")

    # 데이터 로드
    data = load_and_prepare_data()

    if choice == 1:
        # 모델 학습 및 저장
        models, histories, results, scaler = train_and_evaluate_models(data, epochs=3, save=True)
        print("\n✅ 모델 학습 및 저장 완료!")

    elif choice == 2:
        # 저장된 모델 로드 및 평가
        try:
            models, scaler = load_models()
            results = evaluate_loaded_models(models, data, scaler)
            print("\n✅ 저장된 모델 평가 완료!")
        except FileNotFoundError as e:
            print(f"\n❌ 오류: {e}")
            print("먼저 옵션 1을 선택하여 모델을 학습하고 저장하세요.")

    elif choice == 3:
        # 정책 영향 분석 (저장된 모델 사용)
        try:
            models, scaler = load_models()
            analyze_policy_impact(data, models['LSTM'])
            print("\n✅ 정책 영향 분석 완료!")
        except FileNotFoundError as e:
            print(f"\n❌ 오류: {e}")
            print("먼저 옵션 1을 선택하여 모델을 학습하고 저장하세요.")

    elif choice == 4:
        # 통계적 분석
        perform_statistical_analysis(data)
        print("\n✅ 통계적 분석 완료!")

    print("\n" + "="*60)
    print("분석 프로그램 종료")
    print("="*60)

if __name__ == "__main__":
    main()