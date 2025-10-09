# 제1장 종합 시각화
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Setup paths
current_dir = Path(__file__).parent
output_dir = current_dir.parent / 'outputs'
output_dir.mkdir(exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 전체 시각화 대시보드 생성
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Chapter 1: AI and Policy Analysis - Comprehensive Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)

# 1. AI 정책분석 프레임워크 발전 (Timeline)
ax1 = plt.subplot(3, 4, 1)
years = [2020, 2021, 2022, 2023, 2024, 2025]
frameworks = ['Traditional\nPolicy\nAnalysis', 'Data-Driven\nPolicy', 
              'ML-Enhanced\nPolicy', 'AI-Powered\nPolicy', 
              'Causal AI\nPolicy', 'AGI-Ready\nPolicy']
maturity = [3, 4, 5, 7, 8, 9]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(years)))

for i, (year, framework, mat) in enumerate(zip(years, frameworks, maturity)):
    ax1.scatter(year, mat, s=300, c=[colors[i]], alpha=0.6, edgecolors='black', linewidth=2)
    ax1.text(year, mat-0.5, framework, ha='center', fontsize=7)

ax1.plot(years, maturity, 'k--', alpha=0.3)
ax1.set_xlabel('Year')
ax1.set_ylabel('Maturity Level')
ax1.set_title('Evolution of Policy Analysis Frameworks')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 10)

# 2. DPPC 단계별 AI 기술 활용도
ax2 = plt.subplot(3, 4, 2)
stages = ['Agenda\nSetting', 'Policy\nFormulation', 'Decision\nMaking', 
          'Implementation', 'Evaluation']
ai_usage = [65, 78, 85, 72, 90]
traditional = [35, 22, 15, 28, 10]

x = np.arange(len(stages))
width = 0.35

bars1 = ax2.bar(x - width/2, ai_usage, width, label='AI-based', color='#3498db')
bars2 = ax2.bar(x + width/2, traditional, width, label='Traditional', color='#95a5a6')

ax2.set_ylabel('Usage (%)')
ax2.set_title('AI vs Traditional Methods in DPPC')
ax2.set_xticks(x)
ax2.set_xticklabels(stages, fontsize=8)
ax2.legend()
ax2.set_ylim(0, 100)

# 3. 국가별 AI 정책 준비도 (2025)
ax3 = plt.subplot(3, 4, 3)
countries = ['Estonia', 'Singapore', 'Korea', 'USA', 'Japan', 'Germany']
readiness = [93, 91, 85, 88, 83, 80]
colors_country = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#34495e']

bars = ax3.barh(countries, readiness, color=colors_country)
ax3.set_xlabel('AI Policy Readiness Score')
ax3.set_title('Global AI Policy Readiness (2025)')
ax3.set_xlim(0, 100)

for i, (country, score) in enumerate(zip(countries, readiness)):
    ax3.text(score + 1, i, f'{score}%', va='center')

# 4. AI 윤리 원칙 중요도
ax4 = plt.subplot(3, 4, 4, projection='polar')
principles = ['Transparency', 'Fairness', 'Accountability', 
              'Privacy', 'Safety', 'Human Control']
importance = [9.2, 9.5, 8.8, 9.0, 9.3, 8.5]

angles = np.linspace(0, 2 * np.pi, len(principles), endpoint=False).tolist()
importance_plot = importance + importance[:1]
angles += angles[:1]

ax4.plot(angles, importance_plot, 'o-', linewidth=2, color='#e74c3c')
ax4.fill(angles, importance_plot, alpha=0.25, color='#e74c3c')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(principles, size=8)
ax4.set_ylim(0, 10)
ax4.set_title('AI Ethics Principles Importance', y=1.08)
ax4.grid(True)

# 5. 기술별 정책 적용 분야
ax5 = plt.subplot(3, 4, 5)
technologies = ['NLP', 'Computer\nVision', 'Predictive\nModeling', 
                'Causal AI', 'RL']
policy_areas = {
    'Healthcare': [70, 85, 90, 60, 45],
    'Education': [80, 60, 75, 50, 55],
    'Security': [65, 90, 80, 55, 70],
    'Economy': [75, 40, 95, 85, 80]
}

x = np.arange(len(technologies))
width = 0.2
colors_area = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, (area, values) in enumerate(policy_areas.items()):
    offset = width * (i - 1.5)
    ax5.bar(x + offset, values, width, label=area, color=colors_area[i])

ax5.set_ylabel('Application Level (%)')
ax5.set_title('AI Technologies in Policy Areas')
ax5.set_xticks(x)
ax5.set_xticklabels(technologies, fontsize=8)
ax5.legend(loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

# 6. 편향성 검출 메트릭 비교
ax6 = plt.subplot(3, 4, 6)
metrics = ['Demographic\nParity', 'Equalized\nOdds', 'Calibration', 
           'Individual\nFairness']
detection_rate = [85, 78, 72, 65]
false_positive = [12, 8, 15, 20]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, detection_rate, width, label='Detection Rate', 
                color='#2ecc71', alpha=0.8)
bars2 = ax6.bar(x + width/2, false_positive, width, label='False Positive', 
                color='#e74c3c', alpha=0.8)

ax6.set_ylabel('Rate (%)')
ax6.set_title('Bias Detection Metrics Performance')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, fontsize=8)
ax6.legend()
ax6.set_ylim(0, 100)

# 7. AI 규제 프레임워크 비교
ax7 = plt.subplot(3, 4, 7)
regulations = ['EU AI Act', 'Korea AI Law', 'US Framework', 'China Regulation']
aspects = ['Risk-based', 'Transparency', 'Accountability', 'Innovation']

data = np.array([
    [95, 90, 85, 70],  # EU AI Act
    [80, 85, 90, 85],  # Korea AI Law
    [70, 75, 70, 95],  # US Framework
    [85, 60, 80, 75]   # China Regulation
])

im = ax7.imshow(data, cmap='YlOrRd', aspect='auto')
ax7.set_xticks(np.arange(len(aspects)))
ax7.set_yticks(np.arange(len(regulations)))
ax7.set_xticklabels(aspects, fontsize=8)
ax7.set_yticklabels(regulations, fontsize=8)
ax7.set_title('AI Regulation Framework Comparison')

for i in range(len(regulations)):
    for j in range(len(aspects)):
        text = ax7.text(j, i, data[i, j], ha="center", va="center", 
                       color="white" if data[i, j] > 80 else "black", fontsize=8)

# 8. 정책 성과 예측 정확도
ax8 = plt.subplot(3, 4, 8)
methods = ['Linear\nRegression', 'Random\nForest', 'Neural\nNetwork', 
           'Causal\nForest', 'Ensemble']
accuracy = [75, 82, 88, 91, 94]
complexity = [20, 60, 85, 75, 95]

ax8_twin = ax8.twinx()
bars = ax8.bar(methods, accuracy, color='#3498db', alpha=0.7, label='Accuracy')
line = ax8_twin.plot(methods, complexity, 'ro-', linewidth=2, markersize=8, 
                     label='Complexity')

ax8.set_ylabel('Accuracy (%)', color='#3498db')
ax8_twin.set_ylabel('Complexity', color='r')
ax8.set_title('Policy Outcome Prediction Methods')
ax8.set_ylim(0, 100)
ax8_twin.set_ylim(0, 100)
ax8.tick_params(axis='x', rotation=0, labelsize=8)

# Combined legend
lines1, labels1 = ax8.get_legend_handles_labels()
lines2, labels2 = ax8_twin.get_legend_handles_labels()
ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

# 9. AI 도입 장벽 분석
ax9 = plt.subplot(3, 4, 9)
barriers = ['Technical\nCapability', 'Data\nQuality', 'Regulatory\nCompliance', 
            'Cost', 'Public\nTrust', 'Skills\nGap']
severity = [65, 78, 85, 70, 82, 88]
mitigation = [80, 60, 70, 55, 45, 50]

x = np.arange(len(barriers))
width = 0.35

bars1 = ax9.bar(x - width/2, severity, width, label='Severity', color='#e74c3c', alpha=0.8)
bars2 = ax9.bar(x + width/2, mitigation, width, label='Mitigation Effort', 
                color='#3498db', alpha=0.8)

ax9.set_ylabel('Level (%)')
ax9.set_title('AI Adoption Barriers & Mitigation')
ax9.set_xticks(x)
ax9.set_xticklabels(barriers, fontsize=7, rotation=45, ha='right')
ax9.legend(fontsize=8)
ax9.set_ylim(0, 100)

# 10. 실시간 정책 피드백 루프
ax10 = plt.subplot(3, 4, 10)
hours = np.arange(24)
feedback_volume = 50 + 30 * np.sin(np.linspace(0, 2*np.pi, 24)) + \
                  np.random.normal(0, 5, 24)
response_time = 5 + 3 * np.cos(np.linspace(0, 2*np.pi, 24)) + \
                np.random.normal(0, 1, 24)

ax10_twin = ax10.twinx()
ax10.fill_between(hours, feedback_volume, alpha=0.3, color='#3498db', 
                  label='Feedback Volume')
ax10.plot(hours, feedback_volume, 'b-', linewidth=2)
ax10_twin.plot(hours, response_time, 'r-', linewidth=2, label='Response Time')

ax10.set_xlabel('Hour of Day')
ax10.set_ylabel('Feedback Volume', color='b')
ax10_twin.set_ylabel('Response Time (min)', color='r')
ax10.set_title('24-Hour Policy Feedback Cycle')
ax10.grid(True, alpha=0.3)
ax10.set_xlim(0, 23)

# 11. AI 모델 설명가능성 레벨
ax11 = plt.subplot(3, 4, 11)
models = ['Linear\nModel', 'Tree-based', 'Neural Net', 'Deep\nLearning', 'Transformer']
explainability = [95, 75, 40, 25, 20]
performance = [70, 80, 88, 92, 95]

x = np.arange(len(models))
ax11_twin = ax11.twinx()

bars = ax11.bar(x, explainability, color='#2ecc71', alpha=0.7, label='Explainability')
line = ax11_twin.plot(x, performance, 'ko-', linewidth=2, markersize=8, 
                     label='Performance')

ax11.set_ylabel('Explainability (%)', color='#2ecc71')
ax11_twin.set_ylabel('Performance (%)', color='k')
ax11.set_title('Explainability vs Performance Trade-off')
ax11.set_xticks(x)
ax11.set_xticklabels(models, fontsize=8)
ax11.set_ylim(0, 100)
ax11_twin.set_ylim(0, 100)

# Combined legend
lines1, labels1 = ax11.get_legend_handles_labels()
lines2, labels2 = ax11_twin.get_legend_handles_labels()
ax11.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

# 12. 2025년 주요 정책 AI 트렌드
ax12 = plt.subplot(3, 4, 12)
trends = ['Causal AI', 'XAI', 'Federated\nLearning', 'AutoML', 
          'Quantum AI', 'AGI Safety']
adoption = [78, 85, 65, 72, 25, 45]
growth = [25, 20, 30, 22, 45, 35]

# Create bubble chart
for i, (trend, adopt, grow) in enumerate(zip(trends, adoption, growth)):
    size = grow * 20
    color = plt.cm.plasma(adopt/100)
    ax12.scatter(adopt, grow, s=size, c=[color], alpha=0.6, 
                edgecolors='black', linewidth=2)
    ax12.annotate(trend, (adopt, grow), ha='center', va='center', fontsize=7)

ax12.set_xlabel('Current Adoption (%)')
ax12.set_ylabel('Expected Growth (%)')
ax12.set_title('2025 Policy AI Trends')
ax12.grid(True, alpha=0.3)
ax12.set_xlim(0, 100)
ax12.set_ylim(0, 50)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(output_dir / 'chapter01_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== Comprehensive Visualization Complete ===")
print(f"Generated: {output_dir / 'chapter01_comprehensive_dashboard.png'}")
print("Dashboard includes 12 key visualizations covering:")
print("  1. Framework Evolution Timeline")
print("  2. DPPC AI Integration")
print("  3. Global Readiness Comparison")
print("  4. Ethics Principles")
print("  5. Technology Applications")
print("  6. Bias Detection Metrics")
print("  7. Regulatory Frameworks")
print("  8. Prediction Methods")
print("  9. Adoption Barriers")
print(" 10. Feedback Cycles")
print(" 11. Explainability Trade-offs")
print(" 12. 2025 Trends")