"""
Chapter 3: Deep Learning Fundamentals and Policy Time Series Prediction - Korea Electricity Market Data Analysis
Script to load and analyze actual Korea electricity market data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory if not exists
os.makedirs(os.path.join('output'), exist_ok=True)

def load_korea_electricity_data():
    """
    Load and preprocess Korea electricity market data
    """
    print("=" * 60)
    print("Loading Korea Electricity Market Data...")
    print("=" * 60)

    try:
        # Load energy demand data
        demand_df = pd.read_csv(os.path.join('data', 'chapter3_energy_demand.csv'))
        demand_df['timestamp'] = pd.to_datetime(demand_df['timestamp'])

        # Load renewable policy data
        policy_df = pd.read_csv(os.path.join('data', 'chapter3_renewable_policy.csv'))
        policy_df['date'] = pd.to_datetime(policy_df['date'])

        # Load market data
        market_df = pd.read_csv(os.path.join('data', 'chapter3_korea_electricity_market.csv'))
        market_df['date'] = pd.to_datetime(market_df['date'])

        print(f"✅ Energy demand data: {demand_df.shape[0]:,} hourly records")
        print(f"✅ Policy data: {policy_df.shape[0]:,} daily records")
        print(f"✅ Market data: {market_df.shape[0]:,} monthly records")

        return demand_df, policy_df, market_df

    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        print("Please run generate_data.py first to create the data.")
        return None, None, None

def analyze_demand_patterns(demand_df):
    """
    Analyze power demand patterns
    """
    print("\n" + "=" * 60)
    print("Power Demand Pattern Analysis")
    print("=" * 60)

    # Basic statistics
    print("\n📊 Basic Statistics:")
    print(f"Average demand: {demand_df['demand_mw'].mean():,.0f} MW")
    print(f"Maximum demand: {demand_df['demand_mw'].max():,.0f} MW")
    print(f"Minimum demand: {demand_df['demand_mw'].min():,.0f} MW")
    print(f"Standard deviation: {demand_df['demand_mw'].std():,.0f} MW")

    # Seasonal demand analysis
    demand_df['season'] = demand_df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    seasonal_demand = demand_df.groupby('season')['demand_mw'].agg(['mean', 'max', 'min', 'std'])
    print("\n📊 Seasonal Demand Analysis:")
    print(seasonal_demand.round(0))

    # Hourly demand patterns
    hourly_demand = demand_df.groupby('hour')['demand_mw'].mean()

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Average hourly demand
    axes[0, 0].plot(hourly_demand.index, hourly_demand.values, linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Average Demand (MW)')
    axes[0, 0].set_title('Average Power Demand by Hour')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Seasonal demand distribution
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    demand_df['season'] = pd.Categorical(demand_df['season'], categories=season_order, ordered=True)
    demand_df.boxplot(column='demand_mw', by='season', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Season')
    axes[0, 1].set_ylabel('Demand (MW)')
    axes[0, 1].set_title('Power Demand Distribution by Season')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)

    # 3. Weekday demand patterns
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_demand = demand_df.groupby('weekday')['demand_mw'].mean()
    axes[0, 1].get_figure().suptitle('')  # Remove automatic title

    axes[1, 0].bar(range(7), weekday_demand.values, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Weekday')
    axes[1, 0].set_ylabel('Average Demand (MW)')
    axes[1, 0].set_title('Average Power Demand by Weekday')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(weekday_names)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Monthly demand trend
    monthly_demand = demand_df.groupby('month')['demand_mw'].mean()
    axes[1, 1].plot(monthly_demand.index, monthly_demand.values,
                    marker='o', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Demand (MW)')
    axes[1, 1].set_title('Monthly Average Power Demand Trend')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'demand_patterns.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return seasonal_demand, hourly_demand

def analyze_renewable_generation(demand_df):
    """
    Analyze renewable energy generation
    """
    print("\n" + "=" * 60)
    print("Renewable Energy Generation Analysis")
    print("=" * 60)

    # Solar generation analysis
    solar_by_hour = demand_df.groupby('hour')['solar_generation_mw'].mean()
    solar_by_month = demand_df.groupby('month')['solar_generation_mw'].mean()

    # Wind generation analysis
    wind_by_hour = demand_df.groupby('hour')['wind_generation_mw'].mean()
    wind_by_month = demand_df.groupby('month')['wind_generation_mw'].mean()

    print(f"\n☀️ Solar Generation:")
    print(f"Average: {demand_df['solar_generation_mw'].mean():,.0f} MW")
    print(f"Maximum: {demand_df['solar_generation_mw'].max():,.0f} MW")
    print(f"Capacity Factor: {(demand_df['solar_generation_mw'].mean() / 35000 * 100):.1f}%")

    print(f"\n💨 Wind Generation:")
    print(f"Average: {demand_df['wind_generation_mw'].mean():,.0f} MW")
    print(f"Maximum: {demand_df['wind_generation_mw'].max():,.0f} MW")
    print(f"Capacity Factor: {(demand_df['wind_generation_mw'].mean() / 3000 * 100):.1f}%")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Hourly solar generation
    axes[0, 0].plot(solar_by_hour.index, solar_by_hour.values,
                    linewidth=2, color='orange', label='Solar')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Average Generation (MW)')
    axes[0, 0].set_title('Solar Generation Pattern by Hour')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Hourly wind generation
    axes[0, 1].plot(wind_by_hour.index, wind_by_hour.values,
                    linewidth=2, color='blue', label='Wind')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Average Generation (MW)')
    axes[0, 1].set_title('Wind Generation Pattern by Hour')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Monthly renewable energy generation
    axes[1, 0].plot(solar_by_month.index, solar_by_month.values,
                    marker='o', linewidth=2, label='Solar', color='orange')
    axes[1, 0].plot(wind_by_month.index, wind_by_month.values,
                    marker='s', linewidth=2, label='Wind', color='blue')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Generation (MW)')
    axes[1, 0].set_title('Monthly Renewable Energy Generation Trend')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Renewable energy share
    demand_df['renewable_ratio'] = ((demand_df['solar_generation_mw'] +
                                     demand_df['wind_generation_mw']) /
                                    demand_df['demand_mw'] * 100)
    monthly_renewable_ratio = demand_df.groupby('month')['renewable_ratio'].mean()

    axes[1, 1].bar(monthly_renewable_ratio.index, monthly_renewable_ratio.values,
                   color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Renewable Energy Share (%)')
    axes[1, 1].set_title('Monthly Renewable Energy Share')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'renewable_generation.png'), dpi=150, bbox_inches='tight')
    plt.show()

def analyze_policy_impact(policy_df, demand_df):
    """
    Analyze policy impact
    """
    print("\n" + "=" * 60)
    print("Policy Impact Analysis")
    print("=" * 60)

    # Policy phase analysis
    policy_phase_stats = policy_df.groupby('policy_phase').agg({
        'rec_price': 'mean',
        'carbon_price': 'mean',
        'renewable_subsidy': 'mean',
        'renewable_target': 'mean'
    })

    print("\n📊 Policy Phase Metrics:")
    print(policy_phase_stats.round(0))

    # Policy indicators over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. REC price trend
    axes[0, 0].plot(policy_df['date'], policy_df['rec_price'],
                    linewidth=1.5, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('날짜')
    axes[0, 0].set_ylabel('REC Price (KRW/REC)')
    axes[0, 0].set_title('Renewable Energy Certificate (REC) Price Trend')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Carbon price trend
    axes[0, 1].plot(policy_df['date'], policy_df['carbon_price'],
                    linewidth=1.5, color='red', alpha=0.7)
    axes[0, 1].set_xlabel('날짜')
    axes[0, 1].set_ylabel('Carbon Price (KRW/tonCO2)')
    axes[0, 1].set_title('Carbon Price Trend')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Renewable energy target ratio
    axes[1, 0].plot(policy_df['date'], policy_df['renewable_target'],
                    linewidth=2, color='green')
    axes[1, 0].set_xlabel('날짜')
    axes[1, 0].set_ylabel('Renewable Energy Target (%)')
    axes[1, 0].set_title('Renewable Energy Target Ratio Changes')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Cumulative subsidies
    axes[1, 1].plot(policy_df['date'], policy_df['cumulative_subsidy'],
                    linewidth=2, color='purple')
    axes[1, 1].set_xlabel('날짜')
    axes[1, 1].set_ylabel('Cumulative Subsidies (100M KRW)')
    axes[1, 1].set_title('Renewable Energy Cumulative Subsidies')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'policy_impact.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Policy intervention point analysis
    intervention_dates = policy_df[policy_df['policy_intervention'] == 1]['date']
    print(f"\n📌 Major Policy Intervention Points: {len(intervention_dates)} times")
    for date in intervention_dates:
        print(f"   - {date.strftime('%Y-%m-%d')}")

def analyze_market_structure(market_df):
    """
    Analyze electricity market structure
    """
    print("\n" + "=" * 60)
    print("Electricity Market Structure Analysis")
    print("=" * 60)

    # Generation mix analysis
    generation_mix = ['nuclear_pct', 'coal_pct', 'lng_pct', 'renewable_pct', 'other_pct']

    print("\n📊 Annual Average Generation Mix:")
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
                        labels=['Nuclear', 'Coal', 'LNG', 'Renewable', 'Other'],
                        alpha=0.8)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Share (%)')
    axes[0, 0].set_title('Generation Mix Changes')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. SMP 가격 추이
    axes[0, 1].plot(market_df['date'], market_df['smp_price'],
                    marker='o', linewidth=2, color='red', markersize=8)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('SMP (KRW/kWh)')
    axes[0, 1].set_title('System Marginal Price (SMP) Trend')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 예비율 추이
    axes[1, 0].bar(range(len(market_df)), market_df['reserve_margin'],
                   color='blue', alpha=0.7)
    axes[1, 0].axhline(y=15, color='r', linestyle='--', label='Optimal Reserve Margin (15%)')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Reserve Margin (%)')
    axes[1, 0].set_title('Monthly Reserve Margin')
    axes[1, 0].set_xticks(range(len(market_df)))
    axes[1, 0].set_xticklabels([f'M{i+1}' for i in range(len(market_df))])
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
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Capacity (GW)')
    axes[1, 1].set_title('Total Renewable Energy Capacity Growth')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'market_structure.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # SMP and generation mix correlation
    print("\n📊 SMP and Generation Mix Correlation:")
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

    # 1. Daily demand pattern (top left)
    ax1 = plt.subplot(2, 3, 1)
    sample_day = demand_df[demand_df['timestamp'].dt.date == pd.Timestamp('2024-07-15').date()]
    ax1.plot(sample_day['hour'], sample_day['demand_mw'], linewidth=2)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Demand (MW)')
    ax1.set_title('Daily Power Demand Pattern (2024-07-15)')
    ax1.grid(True, alpha=0.3)

    # 2. Monthly demand vs SMP (top center)
    ax2 = plt.subplot(2, 3, 2)
    monthly_demand = demand_df.groupby(demand_df['timestamp'].dt.month)['demand_mw'].mean()
    ax2_twin = ax2.twinx()
    ax2.bar(range(1, 13), monthly_demand.values, alpha=0.7, color='blue', label='Avg Demand')
    ax2_twin.plot(range(1, 13), market_df['smp_price'].values,
                  color='red', marker='o', linewidth=2, label='SMP')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Average Demand (MW)', color='blue')
    ax2_twin.set_ylabel('SMP (KRW/kWh)', color='red')
    ax2.set_title('Monthly Demand vs SMP')
    ax2.grid(True, alpha=0.3)

    # 3. Generation mix (top right)
    ax3 = plt.subplot(2, 3, 3)
    generation_avg = [
        market_df['nuclear_pct'].mean(),
        market_df['coal_pct'].mean(),
        market_df['lng_pct'].mean(),
        market_df['renewable_pct'].mean(),
        market_df['other_pct'].mean()
    ]
    colors = ['yellow', 'gray', 'lightblue', 'green', 'orange']
    ax3.pie(generation_avg, labels=['Nuclear', 'Coal', 'LNG', 'Renewable', 'Other'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Annual Average Generation Mix')

    # 4. Renewable energy generation trend (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    daily_solar = demand_df.groupby(demand_df['timestamp'].dt.date)['solar_generation_mw'].mean()
    daily_wind = demand_df.groupby(demand_df['timestamp'].dt.date)['wind_generation_mw'].mean()
    ax4.plot(daily_solar.index[:30], daily_solar.values[:30], label='Solar', alpha=0.7)
    ax4.plot(daily_wind.index[:30], daily_wind.values[:30], label='Wind', alpha=0.7)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Generation (MW)')
    ax4.set_title('Daily Renewable Energy Generation (January)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Policy indicator changes (bottom center)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(policy_df['date'], policy_df['renewable_target'], linewidth=2, color='green')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Renewable Energy Target (%)')
    ax5.set_title('Renewable Energy Target Ratio Increase')
    ax5.grid(True, alpha=0.3)

    # 6. Demand vs temperature correlation (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    scatter_sample = demand_df.sample(n=1000)
    ax6.scatter(scatter_sample['temperature'], scatter_sample['demand_mw'],
                alpha=0.5, s=10)
    ax6.set_xlabel('Temperature (°C)')
    ax6.set_ylabel('Demand (MW)')
    ax6.set_title('Temperature-Demand Correlation')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('2024년 한국 전력시장 종합 대시보드', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ 분석 완료! 결과는 output 폴더에 저장되었습니다.")

def run_predictive_models(demand_df):
    """
    Run predictive models (LSTM, Transformer, Mamba) and compare performance
    """
    print("\n" + "="*60)
    print("6️⃣ Running Predictive Models Comparison")
    print("="*60)

    try:
        from time_series_models import run_model_comparison

        print("\n🚀 Starting model training and comparison...")
        print("   Models: LSTM, Transformer, Mamba")
        print("   This may take a few minutes...\n")

        # Run model comparison
        comparator, results = run_model_comparison(demand_df, epochs=30)

        print("\n✅ Predictive models comparison completed!")
        print("   Check output/model_comparison.png for visual results")

        return comparator, results

    except ImportError:
        print("\n⚠️ time_series_models.py not found. Skipping predictive models.")
        print("   Please ensure time_series_models.py is in the same directory.")
        return None, None
    except Exception as e:
        print(f"\n❌ Error running predictive models: {e}")
        return None, None

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 60)
    print("Starting Korea Electricity Market Data Analysis")
    print("=" * 60)

    # 데이터 로드
    demand_df, policy_df, market_df = load_korea_electricity_data()

    if demand_df is None:
        print("Data load failed. Exiting program.")
        return

    # Perform analysis
    print("\n1️⃣ Analyzing power demand patterns...")
    seasonal_demand, hourly_demand = analyze_demand_patterns(demand_df)

    print("\n2️⃣ Analyzing renewable energy generation...")
    analyze_renewable_generation(demand_df)

    print("\n3️⃣ Analyzing policy impact...")
    analyze_policy_impact(policy_df, demand_df)

    print("\n4️⃣ Analyzing electricity market structure...")
    analyze_market_structure(market_df)

    print("\n5️⃣ Creating comprehensive report...")
    create_summary_report(demand_df, policy_df, market_df)

    # Run predictive models comparison (automatic)
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS: Deep Learning Models")
    print("="*60)

    # Automatically run predictive models comparison
    comparator, results = run_predictive_models(demand_df)

    print("\n" + "=" * 60)
    print("All analyses completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()