# Example 1.3: Traditional Regression vs Machine Learning Performance Comparison
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('1-1.csv')
print(f"Data loaded from 1-1.csv")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}\n")

# Separate features and target
feature_columns = ['budget', 'population', 'education_level', 'infrastructure', 'previous_performance']
X = data[feature_columns]
y = data['policy_outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Traditional linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# AI-based Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# Compare results
print("=== Traditional vs AI-based Policy Analysis ===\n")
print("Linear Regression (Traditional):")
print(f"  R² Score: {lr_r2:.4f}")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"\nRandom Forest (AI-based):")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"\nPerformance Improvement:")
print(f"  R² Improvement: {(rf_r2 - lr_r2) / lr_r2 * 100:.1f}%")
print(f"  RMSE Reduction: {(lr_rmse - rf_rmse) / lr_rmse * 100:.1f}%")

# 결과 해석
print("\n" + "="*50)
print("📊 결과 해석")
print("="*50)

# R² 점수 해석
if lr_r2 > 0.95 and rf_r2 > 0.95:
    print("\n✅ 모델 성능:")
    print(f"  • 두 모델 모두 매우 높은 설명력을 보임 (R² > 0.95)")
    print(f"  • 선형 회귀: {lr_r2:.1%}의 분산을 설명")
    print(f"  • 랜덤 포레스트: {rf_r2:.1%}의 분산을 설명")
elif rf_r2 > lr_r2:
    print("\n✅ 모델 성능:")
    print(f"  • AI 모델이 전통적 모델보다 우수한 성능")
    print(f"  • 비선형 관계를 더 잘 포착함")
else:
    print("\n✅ 모델 성능:")
    print(f"  • 선형 회귀가 더 나은 성능을 보임")
    print(f"  • 데이터가 선형 관계에 가까움을 시사")

# RMSE 해석
print(f"\n📈 예측 정확도 (RMSE):")
print(f"  • 선형 회귀의 평균 예측 오차: ±{lr_rmse:.1f}")
print(f"  • 랜덤 포레스트의 평균 예측 오차: ±{rf_rmse:.1f}")
if rf_rmse < lr_rmse:
    print(f"  • AI 모델이 {lr_rmse - rf_rmse:.1f} 만큼 더 정확한 예측")
else:
    print(f"  • 선형 모델이 {rf_rmse - lr_rmse:.1f} 만큼 더 정확한 예측")

# 특징 중요도 분석
print(f"\n🔍 주요 정책 요인 (Feature Importance):")
feature_importance_sorted = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_sorted.iterrows():
    feature_name_kr = {
        'budget': '예산',
        'population': '인구',
        'education_level': '교육 수준',
        'infrastructure': '인프라',
        'previous_performance': '이전 성과'
    }
    print(f"  {idx+1}. {feature_name_kr.get(row['feature'], row['feature'])}: {row['importance']:.1%}")

# 종합 평가
print("\n💡 종합 평가:")
if abs(rf_r2 - lr_r2) < 0.01:
    print("  • 이 데이터셋에서는 두 모델의 성능 차이가 미미함")
    print("  • 단순한 선형 모델로도 충분한 예측력을 보임")
    print("  • 계산 효율성을 고려하면 선형 회귀가 더 적합할 수 있음")
elif rf_r2 > lr_r2:
    print("  • AI 기반 모델이 복잡한 패턴을 더 잘 학습함")
    print("  • 비선형 관계와 변수 간 상호작용을 효과적으로 포착")
    print("  • 정책 예측에 머신러닝 활용의 장점을 보여줌")
else:
    print("  • 전통적 통계 모델이 이 경우 더 적합함")
    print("  • 데이터의 선형성이 강하거나 노이즈가 많을 가능성")
    print("  • 과적합 위험 없이 안정적인 예측 제공")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prediction vs actual comparison
ax1 = axes[0]
ax1.scatter(y_test, lr_pred, alpha=0.5, label='Linear Regression', color='blue')
ax1.scatter(y_test, rf_pred, alpha=0.5, label='Random Forest', color='red')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax1.set_xlabel('Actual Policy Outcome')
ax1.set_ylabel('Predicted Policy Outcome')
# ax1.set_title('Prediction Accuracy Comparison')  # Removed per guidelines
ax1.legend()
ax1.grid(True, alpha=0.3)

# Feature importance (Random Forest)
ax2 = axes[1]
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

bars = ax2.bar(range(len(feature_importance)), feature_importance['importance'])
ax2.set_xticks(range(len(feature_importance)))
ax2.set_xticklabels(feature_importance['feature'], rotation=45, ha='right')
ax2.set_ylabel('Feature Importance')
# ax2.set_title('AI Model Feature Importance')  # Removed per guidelines
ax2.grid(True, alpha=0.3)

# Apply color gradient
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.savefig('traditional_vs_ai_comparison.png', dpi=150, bbox_inches='tight')
plt.show()