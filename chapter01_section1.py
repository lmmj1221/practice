# Example 1.1: Korea-Estonia Digital Government Comparison Analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Visualization settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# Data generation: Digital government indicators for Korea and Estonia
data = {
    'Year': [2021, 2022, 2023, 2024, 2025],
    'Korea_Digital_Gov_Index': [72, 75, 78, 82, 85],
    'Estonia_eGov_Satisfaction': [88, 89, 90, 91, 93],
    'Korea_Online_Service': [95, 96, 97, 98, 98],
    'Estonia_Online_Service': [99, 99, 99, 99, 99],
    'Korea_AI_Adoption': [45, 52, 61, 70, 78],
    'Estonia_AI_Adoption': [55, 62, 70, 76, 82]
}

df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

# Create graphs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Digital government index comparison
ax1 = axes[0, 0]
ax1.plot(df.index, df['Korea_Digital_Gov_Index'], 'b-o', label='Korea', linewidth=2, markersize=8)
ax1.plot(df.index, df['Estonia_eGov_Satisfaction'], 'r-s', label='Estonia', linewidth=2, markersize=8)
ax1.set_xlabel('Year')
ax1.set_ylabel('Index Score')
# ax1.set_title('Digital Government Index')  # Removed per guidelines
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(70, 95)

# 2. Online service provision rate
ax2 = axes[0, 1]
width = 0.35
x = np.arange(len(df.index))
bars1 = ax2.bar(x - width/2, df['Korea_Online_Service'], width, label='Korea', color='#3498db')
bars2 = ax2.bar(x + width/2, df['Estonia_Online_Service'], width, label='Estonia', color='#e74c3c')
ax2.set_xlabel('Year')
ax2.set_ylabel('Online Service Rate (%)')
# ax2.set_title('Online Service Provision Rate')  # Removed per guidelines
ax2.set_xticks(x)
ax2.set_xticklabels(df.index)
ax2.legend()
ax2.set_ylim(90, 102)

# 3. AI adoption trends
ax3 = axes[1, 0]
ax3.fill_between(df.index, df['Korea_AI_Adoption'], alpha=0.3, color='blue', label='Korea')
ax3.fill_between(df.index, df['Estonia_AI_Adoption'], alpha=0.3, color='red', label='Estonia')
ax3.plot(df.index, df['Korea_AI_Adoption'], 'b-', linewidth=2)
ax3.plot(df.index, df['Estonia_AI_Adoption'], 'r-', linewidth=2)
ax3.set_xlabel('Year')
ax3.set_ylabel('AI Adoption Rate (%)')
# ax3.set_title('Government AI Adoption Trends')  # Removed per guidelines
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Comprehensive performance radar chart (2025)
ax4 = axes[1, 1]
categories = ['Digital\nIndex', 'Online\nService', 'AI\nAdoption', 'Citizen\nSatisfaction', 'Innovation']
korea_scores = [85, 98, 78, 82, 80]
estonia_scores = [93, 99, 82, 91, 88]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
korea_scores += korea_scores[:1]
estonia_scores += estonia_scores[:1]
angles += angles[:1]

ax4 = plt.subplot(2, 2, 4, projection='polar')
ax4.plot(angles, korea_scores, 'b-', linewidth=2, label='Korea')
ax4.fill(angles, korea_scores, 'blue', alpha=0.25)
ax4.plot(angles, estonia_scores, 'r-', linewidth=2, label='Estonia')
ax4.fill(angles, estonia_scores, 'red', alpha=0.25)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)
ax4.set_ylim(0, 100)
# ax4.set_title('2025 Comprehensive Performance', y=1.08)  # Removed per guidelines
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.grid(True)

plt.tight_layout()
plt.savefig('korea_estonia_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate policy effectiveness index
print("=== Policy Effectiveness Analysis ===\n")
policy_effectiveness = pd.DataFrame({
    'Korea': [85, 98, 78, 82],
    'Estonia': [93, 99, 82, 91]
}, index=['Digital Index', 'Online Service', 'AI Adoption', 'Satisfaction'])

print(policy_effectiveness)
print(f"\nAverage Effectiveness - Korea: {policy_effectiveness['Korea'].mean():.1f}")
print(f"Average Effectiveness - Estonia: {policy_effectiveness['Estonia'].mean():.1f}")
print(f"Gap: {policy_effectiveness['Estonia'].mean() - policy_effectiveness['Korea'].mean():.1f} points")