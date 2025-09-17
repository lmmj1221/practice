# Example 1.3: Traditional Regression vs Machine Learning Performance Comparison
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for policy effect prediction
np.random.seed(42)
n_samples = 1000

# Feature variables (policy input factors)
X = pd.DataFrame({
    'budget': np.random.uniform(100, 1000, n_samples),
    'population': np.random.uniform(10000, 100000, n_samples),
    'education_level': np.random.uniform(0.5, 1.0, n_samples),
    'infrastructure': np.random.uniform(0.3, 0.9, n_samples),
    'previous_performance': np.random.uniform(40, 80, n_samples)
})

# Target variable with nonlinear relationships (policy outcome)
y = (
    0.3 * X['budget'] +
    0.0001 * X['population'] +
    50 * X['education_level']**2 +  # Nonlinear relationship
    30 * X['infrastructure'] * X['education_level'] +  # Interaction
    0.5 * X['previous_performance'] +
    np.random.normal(0, 10, n_samples)  # Noise
)

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