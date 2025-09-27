"""
딥러닝 기초와 정책 시계열 예측 - 교육용 시각화
신경망, RNN, LSTM/GRU 개념을 시각적으로 설명하는 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import warnings
import os

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

def demonstrate_neural_networks():
    """신경망 개념을 시각화하여 설명"""
    print("\n" + "="*50)
    print("신경망 구조 시각화")
    print("="*50)

    # 간단한 신경망 구조 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 퍼셉트론 시각화
    ax = axes[0]
    ax.set_title('단일 퍼셉트론', fontsize=12, pad=20)

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
    ax.text(0.5, -0.1, '기본 뉴런: 입력값들을 가중치와 곱한 후 합산하여\n활성화 함수를 거쳐 출력을 생성',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    # 2. 다층 퍼셉트론 시각화
    ax = axes[1]
    ax.set_title('다층 퍼셉트론 (MLP)', fontsize=12, pad=20)

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
    ax.text(0.5, -0.1, '복잡한 패턴 학습: 여러 층의 뉴런이 연결되어\n비선형적인 관계를 모델링할 수 있음',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, '활성화 함수: 뉴런의 출력을 비선형적으로 변환\nReLU는 음수를 0으로, Sigmoid는 0-1 범위로 압축',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'neural_networks_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 신경망 구조 시각화 완료 ({os.path.join('output', 'neural_networks_demo.png')})")

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
    ax.text(0.5, -0.15, '시계열 데이터는 시간 순서에 따라 관측된 데이터로\n트렌드, 계절성, 노이즈 등의 패턴을 포함',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, '시계열 분해: 원본 데이터를 트렌드(장기 추세),\n계절성(주기적 패턴), 노이즈(무작위 변동)로 분리',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    # 3. 자기상관 함수 (ACF)
    ax = axes[1, 0]
    plot_acf(ts, lags=50, ax=ax)
    ax.set_title('자기상관 함수 (ACF)', fontsize=12)
    ax.text(0.5, -0.15, 'ACF: 현재 시점과 과거 시점 간의 상관관계를 측정\n주기성과 시계열 의존성 파악에 활용',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, '정책 개입의 인과적 효과: 특정 시점의 정책 시행이\n시계열 데이터의 수준 변화를 유발',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'time_series_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 시계열 개념 시각화 완료 ({os.path.join('output', 'time_series_demo.png')})")

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
    ax.text(0.5, -0.1, 'RNN은 이전 시점의 정보를 은닉 상태로 전달하여\n시퀀스 데이터의 시간적 의존성을 학습',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, '기울기 소실: 역전파 시 기울기가 지수적으로 감소하여\n장기 의존성 학습이 어려워지는 현상',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, 'LSTM은 게이트 메커니즘을 통해 장기 의존성을\n효과적으로 학습하여 긴 시퀀스에서도 성능 유지',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'rnn_concepts_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ RNN 개념 시각화 완료 ({os.path.join('output', 'rnn_concepts_demo.png')})")

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

    ax.text(0.5, -0.15, 'LSTM 게이트: Forget(불필요한 정보 제거), Input(새 정보 저장),\nOutput(출력 정보 선택)으로 정보 흐름 제어',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    # 2. GRU vs LSTM 파라미터 수
    ax = axes[0, 1]
    ax.set_title('모델 복잡도 비교', fontsize=12)

    models = ['RNN', 'GRU', 'LSTM']
    params = [100, 300, 400]  # 상대적 파라미터 수

    bars = ax.bar(models, params, color=['gray', 'orange', 'purple'], alpha=0.7)
    ax.set_ylabel('파라미터 수 (상대값)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(0.5, -0.15, 'GRU는 LSTM보다 적은 파라미터로 유사한 성능 달성\n연산 효율성과 학습 속도에서 장점',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, '장기 의존성: LSTM/GRU는 게이트 구조로 정보를\n선택적으로 유지하여 먼 과거 정보도 효과적 활용',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

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
    ax.text(0.5, -0.15, '학습 수렴: LSTM은 안정적, GRU는 빠른 수렴,\nRNN은 상대적으로 높은 손실값에서 정체',
            ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'lstm_gru_demo.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ LSTM/GRU 구조 시각화 완료 ({os.path.join('output', 'lstm_gru_demo.png')})")

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("딥러닝 교육용 시각화 프로그램")
    print("="*60)

    print("\n실행할 시각화를 선택하세요:")
    print("1. 신경망 구조")
    print("2. 시계열 분석 개념")
    print("3. RNN 구조")
    print("4. LSTM/GRU 비교")
    print("5. 전체 실행")

    while True:
        try:
            choice = input("\n선택 (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                choice = int(choice)
                break
            else:
                print("올바른 선택지를 입력하세요 (1-5)")
        except:
            print("잘못된 입력입니다. 다시 시도하세요.")

    if choice == 1:
        demonstrate_neural_networks()
    elif choice == 2:
        demonstrate_time_series_concepts()
    elif choice == 3:
        demonstrate_rnn_concepts()
    elif choice == 4:
        demonstrate_lstm_gru()
    elif choice == 5:
        demonstrate_neural_networks()
        demonstrate_time_series_concepts()
        demonstrate_rnn_concepts()
        demonstrate_lstm_gru()

    print("\n" + "="*60)
    print("교육용 시각화 프로그램 종료")
    print("="*60)

if __name__ == "__main__":
    main()