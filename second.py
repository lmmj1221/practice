"""
제3장: 딥러닝 기초와 정책 시계열 예측 - 한국 전력시장 데이터 분석
실제 한국 전력시장 데이터를 로드하고 분석하는 스크립트
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
try:
    # MacOS
    plt.rcParams['font.family'] = 'AppleGothic'
except:
    try:
        # Windows
        font_path = "C:/Windows/Fonts/malgun.ttf"
        from matplotlib import font_manager
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    except:
        # Fallback to default
        plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory if not exists
os.makedirs(os.path.join('..', 'output'), exist_ok=True)

def load_korea_electricity_data():
    """
    한국 전력시장 데이터 로드 및 전처리
    """
    print("=" * 60)
    print("한국 전력시장 데이터 로드 중...")
    print("=" * 60)

    try:
        # Load energy demand data
        demand_df = pd.read_csv(os.path.join('..', 'data', 'chapter3_energy_demand.csv'))
        demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])

        # Load renewable policy data
        policy_df = pd.read_csv(os.path.join('..', 'data', 'chapter3_renewable_policy.csv'))
        policy_df['date'] = pd.to_datetime(policy_df['date'])

        # Load market data
        market_df = pd.read_csv(os.path.join('..', 'data', 'chapter3_korea_electricity_market.csv'))
        market_df['date'] = pd.to_datetime(market_df['date'])

        print(f"✅ 에너지 수요 데이터: {demand_df.shape[0]:,} 시간별 레코드")
        print(f"✅ 정책 데이터: {policy_df.shape[0]:,} 일별 레코드")
        print(f"✅ 시장 데이터: {market_df.shape[0]:,} 월별 레코드")

        return demand_df, policy_df, market_df

    except FileNotFoundError as e:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        print("먼저 generate_data.py를 실행하여 데이터를 생성해주세요.")
        return None, None, None

def analyze_demand_patterns(demand_df):
    """
    전력 수요 패턴 분석
    """
    print("\n" + "=" * 60)
    print("전력 수요 패턴 분석")
    print("=" * 60)

    # 기본 통계
    print("\n📊 기본 통계:")
    print(f"평균 수요: {demand_df['demand_mw'].mean():,.0f} MW")
    print(f"최대 수요: {demand_df['demand_mw'].max():,.0f} MW")
    print(f"최소 수요: {demand_df['demand_mw'].min():,.0f} MW")
    print(f"표준편차: {demand_df['demand_mw'].std():,.0f} MW")

    # 계절별 수요 분석
    demand_df['season'] = demand_df['month'].map({
        12: '겨울', 1: '겨울', 2: '겨울',
        3: '봄', 4: '봄', 5: '봄',
        6: '여름', 7: '여름', 8: '여름',
        9: '가을', 10: '가을', 11: '가을'
    })

    seasonal_demand = demand_df.groupby('season')['demand_mw'].agg(['mean', 'max', 'min', 'std'])
    print("\n📊 계절별 수요 분석:")
    print(seasonal_demand.round(0))

    # 시간대별 수요 패턴
    hourly_demand = demand_df.groupby('hour')['demand_mw'].mean()

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 시간대별 평균 수요
    axes[0, 0].plot(hourly_demand.index, hourly_demand.values, linewidth=2, color='blue')
    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('평균 수요 (MW)')
    axes[0, 0].set_title('시간대별 평균 전력 수요')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 계절별 수요 분포
    season_order = ['봄', '여름', '가을', '겨울']
    demand_df['season'] = pd.Categorical(demand_df['season'], categories=season_order, ordered=True)
    demand_df.boxplot(column='demand_mw', by='season', ax=axes[0, 1])
    axes[0, 1].set_xlabel('계절')
    axes[0, 1].set_ylabel('수요 (MW)')
    axes[0, 1].set_title('계절별 전력 수요 분포')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)

    # 3. 요일별 수요 패턴
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    weekday_demand = demand_df.groupby('weekday')['demand_mw'].mean()
    axes[0, 1].get_figure().suptitle('')  # Remove automatic title

    axes[1, 0].bar(range(7), weekday_demand.values, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('요일')
    axes[1, 0].set_ylabel('평균 수요 (MW)')
    axes[1, 0].set_title('요일별 평균 전력 수요')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(weekday_names)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 월별 수요 추이
    monthly_demand = demand_df.groupby('month')['demand_mw'].mean()
    axes[1, 1].plot(monthly_demand.index, monthly_demand.values,
                    marker='o', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('평균 수요 (MW)')
    axes[1, 1].set_title('월별 평균 전력 수요 추이')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'demand_patterns.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return seasonal_demand, hourly_demand

def analyze_renewable_generation(demand_df):
    """
    재생에너지 발전 분석
    """
    print("\n" + "=" * 60)
    print("재생에너지 발전 분석")
    print("=" * 60)

    # 태양광 발전 분석
    solar_by_hour = demand_df.groupby('hour')['solar_generation_mw'].mean()
    solar_by_month = demand_df.groupby('month')['solar_generation_mw'].mean()

    # 풍력 발전 분석
    wind_by_hour = demand_df.groupby('hour')['wind_generation_mw'].mean()
    wind_by_month = demand_df.groupby('month')['wind_generation_mw'].mean()

    print(f"\n☀️ 태양광 발전:")
    print(f"평균: {demand_df['solar_generation_mw'].mean():,.0f} MW")
    print(f"최대: {demand_df['solar_generation_mw'].max():,.0f} MW")
    print(f"설비이용률: {(demand_df['solar_generation_mw'].mean() / 35000 * 100):.1f}%")

    print(f"\n💨 풍력 발전:")
    print(f"평균: {demand_df['wind_generation_mw'].mean():,.0f} MW")
    print(f"최대: {demand_df['wind_generation_mw'].max():,.0f} MW")
    print(f"설비이용률: {(demand_df['wind_generation_mw'].mean() / 3000 * 100):.1f}%")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 시간별 태양광 발전
    axes[0, 0].plot(solar_by_hour.index, solar_by_hour.values,
                    linewidth=2, color='orange', label='태양광')
    axes[0, 0].set_xlabel('시간')
    axes[0, 0].set_ylabel('평균 발전량 (MW)')
    axes[0, 0].set_title('시간대별 태양광 발전 패턴')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. 시간별 풍력 발전
    axes[0, 1].plot(wind_by_hour.index, wind_by_hour.values,
                    linewidth=2, color='blue', label='풍력')
    axes[0, 1].set_xlabel('시간')
    axes[0, 1].set_ylabel('평균 발전량 (MW)')
    axes[0, 1].set_title('시간대별 풍력 발전 패턴')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. 월별 재생에너지 발전
    axes[1, 0].plot(solar_by_month.index, solar_by_month.values,
                    marker='o', linewidth=2, label='태양광', color='orange')
    axes[1, 0].plot(wind_by_month.index, wind_by_month.values,
                    marker='s', linewidth=2, label='풍력', color='blue')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].set_ylabel('평균 발전량 (MW)')
    axes[1, 0].set_title('월별 재생에너지 발전 추이')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. 재생에너지 비중
    demand_df['renewable_ratio'] = ((demand_df['solar_generation_mw'] +
                                     demand_df['wind_generation_mw']) /
                                    demand_df['demand_mw'] * 100)
    monthly_renewable_ratio = demand_df.groupby('month')['renewable_ratio'].mean()

    axes[1, 1].bar(monthly_renewable_ratio.index, monthly_renewable_ratio.values,
                   color='green', alpha=0.7)
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('재생에너지 비중 (%)')
    axes[1, 1].set_title('월별 재생에너지 비중')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'renewable_generation.png'), dpi=150, bbox_inches='tight')
    plt.show()

def analyze_policy_impact(policy_df, demand_df):
    """
    정책 영향 분석
    """
    print("\n" + "=" * 60)
    print("정책 영향 분석")
    print("=" * 60)

    # 정책 단계별 분석
    policy_phase_stats = policy_df.groupby('policy_phase').agg({
        'rec_price': 'mean',
        'carbon_price': 'mean',
        'renewable_subsidy': 'mean',
        'renewable_target': 'mean'
    })

    print("\n📊 정책 단계별 지표:")
    print(policy_phase_stats.round(0))

    # 시간에 따른 정책 지표 변화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. REC 가격 추이
    axes[0, 0].plot(policy_df['date'], policy_df['rec_price'],
                    linewidth=1.5, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('날짜')
    axes[0, 0].set_ylabel('REC 가격 (원/REC)')
    axes[0, 0].set_title('신재생에너지 공급인증서(REC) 가격 추이')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 탄소 가격 추이
    axes[0, 1].plot(policy_df['date'], policy_df['carbon_price'],
                    linewidth=1.5, color='red', alpha=0.7)
    axes[0, 1].set_xlabel('날짜')
    axes[0, 1].set_ylabel('탄소 가격 (원/톤CO2)')
    axes[0, 1].set_title('탄소 가격 추이')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 재생에너지 목표 비율
    axes[1, 0].plot(policy_df['date'], policy_df['renewable_target'],
                    linewidth=2, color='green')
    axes[1, 0].set_xlabel('날짜')
    axes[1, 0].set_ylabel('재생에너지 목표 (%)')
    axes[1, 0].set_title('재생에너지 목표 비율 변화')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 누적 보조금
    axes[1, 1].plot(policy_df['date'], policy_df['cumulative_subsidy'],
                    linewidth=2, color='purple')
    axes[1, 1].set_xlabel('날짜')
    axes[1, 1].set_ylabel('누적 보조금 (억원)')
    axes[1, 1].set_title('재생에너지 누적 보조금')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'policy_impact.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 정책 개입 시점 분석
    intervention_dates = policy_df[policy_df['policy_intervention'] == 1]['date']
    print(f"\n📌 주요 정책 개입 시점: {len(intervention_dates)}회")
    for date in intervention_dates:
        print(f"   - {date.strftime('%Y-%m-%d')}")

def analyze_market_structure(market_df):
    """
    전력시장 구조 분석
    """
    print("\n" + "=" * 60)
    print("전력시장 구조 분석")
    print("=" * 60)

    # 발전원별 비중 분석
    generation_mix = ['nuclear_pct', 'coal_pct', 'lng_pct', 'renewable_pct', 'other_pct']

    print("\n📊 연평균 발전원별 비중:")
    for source in generation_mix:
        avg_pct = market_df[source].mean()
        print(f"   {source.replace('_pct', '').upper()}: {avg_pct:.1f}%")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 발전원별 비중 추이 (Stacked Area Chart)
    axes[0, 0].stackplot(market_df['date'],
                        market_df['nuclear_pct'],
                        market_df['coal_pct'],
                        market_df['lng_pct'],
                        market_df['renewable_pct'],
                        market_df['other_pct'],
                        labels=['원자력', '석탄', 'LNG', '신재생', '기타'],
                        alpha=0.8)
    axes[0, 0].set_xlabel('날짜')
    axes[0, 0].set_ylabel('비중 (%)')
    axes[0, 0].set_title('발전원별 비중 변화')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. SMP 가격 추이
    axes[0, 1].plot(market_df['date'], market_df['smp_price'],
                    marker='o', linewidth=2, color='red', markersize=8)
    axes[0, 1].set_xlabel('날짜')
    axes[0, 1].set_ylabel('SMP (원/kWh)')
    axes[0, 1].set_title('시장한계가격(SMP) 추이')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 예비율 추이
    axes[1, 0].bar(range(len(market_df)), market_df['reserve_margin'],
                   color='blue', alpha=0.7)
    axes[1, 0].axhline(y=15, color='r', linestyle='--', label='적정 예비율 (15%)')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].set_ylabel('예비율 (%)')
    axes[1, 0].set_title('월별 예비율')
    axes[1, 0].set_xticks(range(len(market_df)))
    axes[1, 0].set_xticklabels([f'{i+1}월' for i in range(len(market_df))])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 재생에너지 설비용량 증가
    renewable_capacity = (market_df['solar_capacity_mw'] +
                         market_df['wind_capacity_mw'] +
                         market_df['hydro_capacity_mw'] +
                         market_df['bio_capacity_mw'] +
                         market_df['fuel_cell_capacity_mw'])

    axes[1, 1].plot(market_df['date'], renewable_capacity / 1000,
                    marker='s', linewidth=2, color='green', markersize=6)
    axes[1, 1].set_xlabel('날짜')
    axes[1, 1].set_ylabel('설비용량 (GW)')
    axes[1, 1].set_title('재생에너지 총 설비용량 증가')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'market_structure.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # SMP와 발전원별 상관관계
    print("\n📊 SMP와 발전원별 비중 상관관계:")
    for source in generation_mix:
        corr = market_df['smp_price'].corr(market_df[source])
        print(f"   {source.replace('_pct', '').upper()}: {corr:.3f}")

def create_summary_report(demand_df, policy_df, market_df):
    """
    종합 분석 보고서 생성
    """
    print("\n" + "=" * 60)
    print("2024년 한국 전력시장 종합 분석 보고서")
    print("=" * 60)

    # 연간 주요 지표
    print("\n📈 2024년 연간 주요 지표:")
    print(f"총 전력수요: {demand_df['demand_mw'].sum() / 1000:,.0f} GWh")
    print(f"최대 전력수요: {demand_df['demand_mw'].max():,.0f} MW")
    print(f"평균 전력수요: {demand_df['demand_mw'].mean():,.0f} MW")

    # 재생에너지 발전
    total_renewable = (demand_df['solar_generation_mw'].sum() +
                      demand_df['wind_generation_mw'].sum()) / 1000
    print(f"\n🌱 재생에너지 발전량: {total_renewable:,.0f} GWh")
    print(f"재생에너지 평균 비중: {((demand_df['solar_generation_mw'] + demand_df['wind_generation_mw']) / demand_df['demand_mw']).mean() * 100:.1f}%")

    # 정책 지표
    print(f"\n📋 정책 지표:")
    print(f"평균 REC 가격: {policy_df['rec_price'].mean():,.0f} 원/REC")
    print(f"평균 탄소 가격: {policy_df['carbon_price'].mean():,.0f} 원/톤CO2")
    print(f"총 보조금: {policy_df['renewable_subsidy'].sum():,.0f} 억원")

    # 시장 지표
    print(f"\n💰 시장 지표:")
    print(f"평균 SMP: {market_df['smp_price'].mean():.1f} 원/kWh")
    print(f"평균 예비율: {market_df['reserve_margin'].mean():.1f}%")

    # 종합 시각화
    fig = plt.figure(figsize=(16, 10))

    # 1. 일일 수요 패턴 (상단 좌측)
    ax1 = plt.subplot(2, 3, 1)
    sample_day = demand_df[demand_df['timestamp'].dt.date == pd.Timestamp('2024-07-15').date()]
    ax1.plot(sample_day['hour'], sample_day['demand_mw'], linewidth=2)
    ax1.set_xlabel('시간')
    ax1.set_ylabel('수요 (MW)')
    ax1.set_title('일일 전력수요 패턴 (2024-07-15)')
    ax1.grid(True, alpha=0.3)

    # 2. 월별 수요 vs SMP (상단 중앙)
    ax2 = plt.subplot(2, 3, 2)
    monthly_demand = demand_df.groupby(demand_df['timestamp'].dt.month)['demand_mw'].mean()
    ax2_twin = ax2.twinx()
    ax2.bar(range(1, 13), monthly_demand.values, alpha=0.7, color='blue', label='평균수요')
    ax2_twin.plot(range(1, 13), market_df['smp_price'].values,
                  color='red', marker='o', linewidth=2, label='SMP')
    ax2.set_xlabel('월')
    ax2.set_ylabel('평균 수요 (MW)', color='blue')
    ax2_twin.set_ylabel('SMP (원/kWh)', color='red')
    ax2.set_title('월별 수요 vs SMP')
    ax2.grid(True, alpha=0.3)

    # 3. 발전원 구성 (상단 우측)
    ax3 = plt.subplot(2, 3, 3)
    generation_avg = [
        market_df['nuclear_pct'].mean(),
        market_df['coal_pct'].mean(),
        market_df['lng_pct'].mean(),
        market_df['renewable_pct'].mean(),
        market_df['other_pct'].mean()
    ]
    colors = ['yellow', 'gray', 'lightblue', 'green', 'orange']
    ax3.pie(generation_avg, labels=['원자력', '석탄', 'LNG', '신재생', '기타'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('연평균 발전원 구성')

    # 4. 재생에너지 발전 추이 (하단 좌측)
    ax4 = plt.subplot(2, 3, 4)
    daily_solar = demand_df.groupby(demand_df['timestamp'].dt.date)['solar_generation_mw'].mean()
    daily_wind = demand_df.groupby(demand_df['timestamp'].dt.date)['wind_generation_mw'].mean()
    ax4.plot(daily_solar.index[:30], daily_solar.values[:30], label='태양광', alpha=0.7)
    ax4.plot(daily_wind.index[:30], daily_wind.values[:30], label='풍력', alpha=0.7)
    ax4.set_xlabel('날짜')
    ax4.set_ylabel('발전량 (MW)')
    ax4.set_title('일별 재생에너지 발전 (1월)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. 정책 지표 변화 (하단 중앙)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(policy_df['date'], policy_df['renewable_target'], linewidth=2, color='green')
    ax5.set_xlabel('날짜')
    ax5.set_ylabel('재생에너지 목표 (%)')
    ax5.set_title('재생에너지 목표 비율 증가')
    ax5.grid(True, alpha=0.3)

    # 6. 수요 vs 온도 상관관계 (하단 우측)
    ax6 = plt.subplot(2, 3, 6)
    scatter_sample = demand_df.sample(n=1000)
    ax6.scatter(scatter_sample['temperature'], scatter_sample['demand_mw'],
                alpha=0.5, s=10)
    ax6.set_xlabel('온도 (°C)')
    ax6.set_ylabel('수요 (MW)')
    ax6.set_title('온도-수요 상관관계')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('2024년 한국 전력시장 종합 대시보드', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join('..', 'output', 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ 분석 완료! 결과는 output 폴더에 저장되었습니다.")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("한국 전력시장 데이터 분석 시작")
    print("=" * 60)

    # 데이터 로드
    demand_df, policy_df, market_df = load_korea_electricity_data()

    if demand_df is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return

    # 분석 수행
    print("\n1️⃣ 전력 수요 패턴 분석 중...")
    seasonal_demand, hourly_demand = analyze_demand_patterns(demand_df)

    print("\n2️⃣ 재생에너지 발전 분석 중...")
    analyze_renewable_generation(demand_df)

    print("\n3️⃣ 정책 영향 분석 중...")
    analyze_policy_impact(policy_df, demand_df)

    print("\n4️⃣ 전력시장 구조 분석 중...")
    analyze_market_structure(market_df)

    print("\n5️⃣ 종합 보고서 생성 중...")
    create_summary_report(demand_df, policy_df, market_df)

    print("\n" + "=" * 60)
    print("모든 분석이 완료되었습니다!")
    print("=" * 60)

if __name__ == "__main__":
    main()