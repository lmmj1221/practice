# Example 1.3: Traditional Regression vs Machine Learning Performance Comparison
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup paths
current_dir = Path(__file__).parent
output_dir = current_dir.parent / 'outputs'
output_dir.mkdir(exist_ok=True)

# Generate synthetic data for policy effect prediction
np.random.seed(42)
n_samples = 2000

# Feature variables (policy input factors) - normalized values
X = pd.DataFrame({
    'budget': np.random.uniform(0, 1, n_samples),
    'population': np.random.uniform(0, 1, n_samples),
    'education_level': np.random.uniform(0, 1, n_samples),
    'infrastructure': np.random.uniform(0, 1, n_samples),
    'previous_performance': np.random.uniform(0, 1, n_samples)
})

# Target variable with strong nonlinear relationships and interaction effects
y = (
    # Linear effects (weak)
    2 * X['budget'] +
    1 * X['population'] +

    # Strong nonlinear effects
    100 * X['education_level']**3 +  # Cubic effect
    50 * X['infrastructure']**2 +  # Quadratic effect

    # Interaction effects
    80 * X['infrastructure'] * X['education_level'] +
    40 * X['budget'] * X['education_level']**2 +

    # Complex nonlinear patterns
    20 * np.sin(X['budget'] * 2 * np.pi) +  # Periodic pattern
    15 * np.cos(X['infrastructure'] * 2 * np.pi) +

    # Threshold effect
    30 * (X['education_level'] > 0.7).astype(int) +

    # Previous performance nonlinear effect
    10 * np.log(X['previous_performance'] + 0.1) +

    # Noise
    np.random.normal(0, 5, n_samples)
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Traditional linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# AI-based Random Forest (with optimized parameters)
rf_model = RandomForestRegressor(
    n_estimators=200,  # More trees
    max_depth=15,  # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# Compare results
print("=" * 70)
print("전통적 회귀분석 vs 머신러닝 성능 비교 (강한 비선형 데이터)")
print("=" * 70)
print()
print("데이터 특성:")
print("- 3차 다항식 효과 (education_level³)")
print("- 다중 상호작용 효과")
print("- 주기적 패턴 (sin, cos)")
print("- 임계값 효과 (education > 0.7)")
print("- 로그 변환 효과")
print()
print("**실제 실행 결과**")
print()
print("Linear Regression (Traditional):")
print(f"  R² Score: {lr_r2:.4f}")
print(f"  RMSE: {lr_rmse:.2f}")
print(f"\nRandom Forest (AI-based):")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.2f}")
print(f"\nPerformance Improvement:")
r2_improvement = (rf_r2 - lr_r2) / abs(lr_r2) * 100 if lr_r2 != 0 else 0
rmse_reduction = (lr_rmse - rf_rmse) / lr_rmse * 100
print(f"  R² Improvement: {r2_improvement:.1f}%")
print(f"  RMSE Reduction: {rmse_reduction:.1f}%")

# Feature importance analysis
print("\n**Random Forest 특징 중요도 분석**")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"- {row['feature']}: {row['importance']*100:.1f}%")

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
plt.savefig(output_dir / 'traditional_vs_ai_comparison.png', dpi=150, bbox_inches='tight')
plt.show()