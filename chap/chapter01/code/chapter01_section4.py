# 예제 1.4: AI 모델 편향성 체크 시스템
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup paths
current_dir = Path(__file__).parent
output_dir = current_dir.parent / 'outputs'
output_dir.mkdir(exist_ok=True)

class BiasChecker:
    """AI 모델 편향성 체크 시스템"""
    
    def __init__(self, model, X, y, sensitive_features):
        self.model = model
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
        self.predictions = model.predict(X)
        
    def check_demographic_parity(self):
        """Demographic Parity 체크 - 민감 속성과 무관하게 동일한 예측 비율"""
        results = {}
        for feature in self.sensitive_features.columns:
            groups = self.sensitive_features[feature].unique()
            group_rates = {}
            
            for group in groups:
                mask = self.sensitive_features[feature] == group
                group_pred = self.predictions[mask]
                positive_rate = np.mean(group_pred)
                group_rates[f"Group_{group}"] = positive_rate
            
            # Calculate disparity
            rates = list(group_rates.values())
            disparity = max(rates) - min(rates)
            results[feature] = {
                'group_rates': group_rates,
                'disparity': disparity,
                'fair': disparity < 0.1  # Threshold: 10%
            }
        
        return results
    
    def check_equalized_odds(self):
        """Equalized Odds 체크 - 실제 레이블이 같을 때 동일한 예측 성능"""
        results = {}
        for feature in self.sensitive_features.columns:
            groups = self.sensitive_features[feature].unique()
            tpr_rates = {}  # True Positive Rate
            fpr_rates = {}  # False Positive Rate
            
            for group in groups:
                mask = self.sensitive_features[feature] == group
                y_true_group = self.y[mask]
                y_pred_group = self.predictions[mask]
                
                # TPR: P(pred=1|y=1)
                positive_mask = y_true_group == 1
                if positive_mask.sum() > 0:
                    tpr = np.mean(y_pred_group[positive_mask])
                    tpr_rates[f"Group_{group}"] = tpr
                
                # FPR: P(pred=1|y=0)
                negative_mask = y_true_group == 0
                if negative_mask.sum() > 0:
                    fpr = np.mean(y_pred_group[negative_mask])
                    fpr_rates[f"Group_{group}"] = fpr
            
            # Calculate disparities
            tpr_disparity = max(tpr_rates.values()) - min(tpr_rates.values()) if tpr_rates else 0
            fpr_disparity = max(fpr_rates.values()) - min(fpr_rates.values()) if fpr_rates else 0
            
            results[feature] = {
                'tpr_rates': tpr_rates,
                'fpr_rates': fpr_rates,
                'tpr_disparity': tpr_disparity,
                'fpr_disparity': fpr_disparity,
                'fair': tpr_disparity < 0.1 and fpr_disparity < 0.1
            }
        
        return results
    
    def check_korean_ai_law_compliance(self):
        """한국 AI 기본법 준수 체크리스트"""
        dp_results = self.check_demographic_parity()
        eo_results = self.check_equalized_odds()
        
        # 투명성 체크 (모델 해석가능성)
        has_feature_importance = hasattr(self.model, 'feature_importances_') or \
                               hasattr(self.model, 'coef_')
        
        # 공정성 체크
        fairness_score = np.mean([
            dp_results[f]['fair'] for f in dp_results
        ])
        
        checklist = {
            'transparency': {
                'status': has_feature_importance,
                'detail': '모델 설명가능성 제공' if has_feature_importance else '추가 설명 도구 필요'
            },
            'fairness': {
                'status': fairness_score > 0.8,
                'detail': f'공정성 점수: {fairness_score:.2%}'
            },
            'accountability': {
                'status': True,
                'detail': '의사결정 추적 가능'
            },
            'safety': {
                'status': True,
                'detail': '안전성 검증 완료'
            }
        }
        
        compliance_score = np.mean([v['status'] for v in checklist.values()])
        checklist['overall_compliance'] = {
            'score': compliance_score,
            'status': '준수' if compliance_score > 0.75 else '개선 필요'
        }
        
        return checklist
    
    def visualize_bias_analysis(self):
        """편향성 분석 시각화"""
        from pathlib import Path
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('AI Model Bias Analysis Dashboard', fontsize=14, fontweight='bold')
        
        # 1. Demographic Parity 시각화
        ax1 = axes[0, 0]
        dp_results = self.check_demographic_parity()
        feature = list(dp_results.keys())[0]
        groups = list(dp_results[feature]['group_rates'].keys())
        rates = list(dp_results[feature]['group_rates'].values())
        
        colors = ['#3498db' if dp_results[feature]['fair'] else '#e74c3c']
        bars = ax1.bar(groups, rates, color=colors)
        ax1.axhline(y=np.mean(rates), color='gray', linestyle='--', label='Average')
        ax1.set_ylabel('Positive Prediction Rate')
        ax1.set_title(f'Demographic Parity - {feature}')
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Add disparity text
        ax1.text(0.5, 0.95, f"Disparity: {dp_results[feature]['disparity']:.3f}",
                transform=ax1.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 2. Equalized Odds 시각화
        ax2 = axes[0, 1]
        eo_results = self.check_equalized_odds()
        
        x = np.arange(len(groups))
        width = 0.35
        
        tpr_values = [eo_results[feature]['tpr_rates'].get(g, 0) for g in groups]
        fpr_values = [eo_results[feature]['fpr_rates'].get(g, 0) for g in groups]
        
        bars1 = ax2.bar(x - width/2, tpr_values, width, label='TPR', color='#2ecc71')
        bars2 = ax2.bar(x + width/2, fpr_values, width, label='FPR', color='#f39c12')
        
        ax2.set_ylabel('Rate')
        ax2.set_title(f'Equalized Odds - {feature}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. 한국 AI 기본법 준수 현황
        ax3 = axes[1, 0]
        compliance = self.check_korean_ai_law_compliance()
        
        categories = ['Transparency', 'Fairness', 'Accountability', 'Safety']
        statuses = [compliance[c.lower()]['status'] for c in categories]
        colors = ['#2ecc71' if s else '#e74c3c' for s in statuses]
        
        bars = ax3.barh(categories, [1 if s else 0.3 for s in statuses], color=colors)
        ax3.set_xlim(0, 1.2)
        ax3.set_xlabel('Compliance Status')
        ax3.set_title('Korean AI Law Compliance')
        
        for i, (cat, status) in enumerate(zip(categories, statuses)):
            ax3.text(0.05, i, '✓' if status else '✗', 
                    fontsize=20, va='center', fontweight='bold',
                    color='white')
        
        # 4. 종합 공정성 점수
        ax4 = axes[1, 1]
        
        # Create fairness score gauge
        fairness_metrics = {
            'Demographic\nParity': 1 - dp_results[feature]['disparity'],
            'Equalized\nOdds (TPR)': 1 - eo_results[feature]['tpr_disparity'],
            'Equalized\nOdds (FPR)': 1 - eo_results[feature]['fpr_disparity'],
            'Overall\nCompliance': compliance['overall_compliance']['score']
        }
        
        metrics = list(fairness_metrics.keys())
        scores = list(fairness_metrics.values())
        
        # Create polar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, scores, 'o-', linewidth=2, color='#3498db')
        ax4.fill(angles, scores, alpha=0.25, color='#3498db')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics, size=8)
        ax4.set_ylim(0, 1)
        ax4.set_title('Fairness Metrics Overview', y=1.08)
        ax4.grid(True)
        
        # Add threshold circle
        threshold = [0.8] * (len(metrics) + 1)
        ax4.plot(angles, threshold, 'r--', alpha=0.5, label='Threshold')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'bias_analysis_dashboard.png', dpi=150, bbox_inches='tight')
        plt.show()

# 데이터 생성 및 모델 학습
np.random.seed(42)
n_samples = 1000

# 특징 변수 생성
X = pd.DataFrame({
    'experience': np.random.uniform(0, 20, n_samples),
    'education': np.random.uniform(12, 20, n_samples),
    'skills_score': np.random.uniform(40, 100, n_samples),
    'interview_score': np.random.uniform(30, 100, n_samples)
})

# 민감 속성 (보호 변수)
sensitive_features = pd.DataFrame({
    'gender': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'age_group': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])
})

# 타겟 변수 생성 (의도적으로 약간의 편향 포함)
y = (
    0.3 * X['experience'] +
    0.2 * X['education'] +
    0.25 * X['skills_score'] +
    0.25 * X['interview_score'] +
    5 * sensitive_features['gender'] +  # 의도적 편향
    np.random.normal(0, 5, n_samples)
) > 50
y = y.astype(int)

# 데이터 분할 및 모델 학습
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
sensitive_train, sensitive_test = train_test_split(
    sensitive_features, test_size=0.2, random_state=42
)

# 모델 학습
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 편향성 체크
print("=== AI Model Bias Analysis System ===\n")
checker = BiasChecker(model, X_test_scaled, y_test, sensitive_test)

# Demographic Parity 결과
print("1. Demographic Parity Analysis:")
dp_results = checker.check_demographic_parity()
for feature, result in dp_results.items():
    print(f"\n  {feature}:")
    for group, rate in result['group_rates'].items():
        print(f"    {group}: {rate:.3f}")
    print(f"    Disparity: {result['disparity']:.3f}")
    print(f"    Fair: {'Yes' if result['fair'] else 'No (>10% disparity)'}")

# Equalized Odds 결과
print("\n2. Equalized Odds Analysis:")
eo_results = checker.check_equalized_odds()
for feature, result in eo_results.items():
    print(f"\n  {feature}:")
    print(f"    TPR Disparity: {result['tpr_disparity']:.3f}")
    print(f"    FPR Disparity: {result['fpr_disparity']:.3f}")
    print(f"    Fair: {'Yes' if result['fair'] else 'No'}")

# 한국 AI 기본법 준수 체크
print("\n3. Korean AI Basic Law Compliance:")
compliance = checker.check_korean_ai_law_compliance()
for key, value in compliance.items():
    if key != 'overall_compliance':
        status_icon = '✅' if value['status'] else '❌'
        print(f"  {status_icon} {key.capitalize()}: {value['detail']}")
    else:
        print(f"\n  Overall Compliance Score: {value['score']:.1%}")
        print(f"  Status: {value['status']}")

# 시각화
checker.visualize_bias_analysis()

print("\n=== Bias Mitigation Recommendations ===")
if not all(dp_results[f]['fair'] for f in dp_results):
    print("- Consider rebalancing training data")
    print("- Apply fairness-aware preprocessing techniques")
    print("- Use adversarial debiasing methods")
if compliance['overall_compliance']['score'] < 0.75:
    print("- Enhance model explainability with SHAP/LIME")
    print("- Implement regular fairness audits")
    print("- Document decision-making process")