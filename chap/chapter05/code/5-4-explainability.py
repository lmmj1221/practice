#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제5장: 설명가능한 AI 구현
SHAP과 LIME을 활용한 모델 설명가능성 확보
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class ExplainabilityImplementation:
    """설명가능한 AI 구현 클래스"""

    def __init__(self, random_state=42):
        """
        설명가능성 구현 초기화

        Parameters:
        random_state (int): 재현성을 위한 시드
        """
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.scaler = StandardScaler()

    def generate_policy_explanation_data(self, n_samples=1000):
        """
        정책 설명을 위한 시뮬레이션 데이터 생성
        ※ 본 데이터는 교육 목적의 시뮬레이션 데이터입니다

        Parameters:
        n_samples (int): 샘플 수

        Returns:
        tuple: (X, y, feature_names) 특성, 타겟, 특성명
        """
        np.random.seed(self.random_state)

        # 정책 관련 특성 정의
        feature_names = [
            '경제성장률(%)', '실업률(%)', '인플레이션율(%)',
            '정부지출비율(%)', '교육예산비율(%)', '인프라투자비율(%)',
            '혁신투자비율(%)', '사회보장비율(%)', '인구증가율(%)', '도시화율(%)'
        ]

        self.feature_names = feature_names
        n_features = len(feature_names)

        # 특성별 데이터 생성 (실제 정책 데이터 분포 모방)
        X = np.zeros((n_samples, n_features))

        # 경제성장률: 정규분포 (평균 2.5%, 표준편차 1.5%)
        X[:, 0] = np.random.normal(2.5, 1.5, n_samples)

        # 실업률: 로그정규분포 (평균 4%, 최소 1%)
        X[:, 1] = np.maximum(1.0, np.random.lognormal(1.2, 0.3, n_samples))

        # 인플레이션율: 정규분포 (평균 2%, 표준편차 1%)
        X[:, 2] = np.random.normal(2.0, 1.0, n_samples)

        # 정부지출비율: 균등분포 (15-30%)
        X[:, 3] = np.random.uniform(15, 30, n_samples)

        # 교육예산비율: 감마분포 (평균 4%)
        X[:, 4] = np.random.gamma(2, 2, n_samples)

        # 인프라투자비율: 지수분포 (평균 3%)
        X[:, 5] = np.random.exponential(3, n_samples)

        # 혁신투자비율: 베타분포 (0-5% 범위)
        X[:, 6] = np.random.beta(2, 5, n_samples) * 5

        # 사회보장비율: 정규분포 (평균 8%, 표준편차 2%)
        X[:, 7] = np.random.normal(8, 2, n_samples)

        # 인구증가율: 정규분포 (평균 0.5%, 표준편차 0.3%)
        X[:, 8] = np.random.normal(0.5, 0.3, n_samples)

        # 도시화율: 베타분포 (40-90% 범위)
        X[:, 9] = 40 + np.random.beta(2, 2, n_samples) * 50

        # 복잡한 정책 효과 함수 정의
        # 정책만족도 = f(경제적 요인, 사회적 요인, 구조적 요인)
        y = self._calculate_policy_satisfaction(X)

        # DataFrame 생성
        X_df = pd.DataFrame(X, columns=feature_names)

        print(f"✅ 정책 설명용 데이터 생성 완료")
        print(f"   - 샘플 수: {n_samples}")
        print(f"   - 특성 수: {n_features}")
        print(f"   - 정책만족도 범위: [{y.min():.2f}, {y.max():.2f}]")

        return X_df, y, feature_names

    def _calculate_policy_satisfaction(self, X):
        """
        복잡한 정책 만족도 계산 함수

        Parameters:
        X (array): 입력 특성

        Returns:
        array: 정책 만족도 점수
        """
        # 경제적 요인 (가중 평균)
        economic_factor = (
            0.4 * X[:, 0] +                    # 경제성장률 (+)
            -0.3 * X[:, 1] +                   # 실업률 (-)
            -0.2 * np.power(X[:, 2], 2) +      # 인플레이션율 제곱 (-)
            0.1 * X[:, 3]                      # 정부지출비율 (+)
        )

        # 사회적 요인
        social_factor = (
            0.3 * X[:, 4] +                    # 교육예산비율 (+)
            0.2 * np.log(X[:, 5] + 1) +        # 인프라투자비율 로그 (+)
            0.4 * X[:, 6] +                    # 혁신투자비율 (+)
            0.25 * X[:, 7]                     # 사회보장비율 (+)
        )

        # 구조적 요인
        structural_factor = (
            0.2 * X[:, 8] +                    # 인구증가율 (+)
            0.1 * np.power(X[:, 9] / 100, 0.5) # 도시화율 제곱근 (+)
        )

        # 상호작용 효과
        interaction_effects = (
            0.05 * X[:, 0] * X[:, 4] +         # 경제성장-교육 상호작용
            -0.03 * X[:, 1] * X[:, 2] +        # 실업-인플레이션 상호작용
            0.02 * X[:, 3] * X[:, 7] +         # 정부지출-사회보장 상호작용
            0.01 * X[:, 5] * X[:, 9] / 100     # 인프라-도시화 상호작용
        )

        # 최종 정책 만족도 (0-100 점 스케일)
        satisfaction = (
            50 +                               # 기준점
            economic_factor +
            social_factor +
            structural_factor +
            interaction_effects +
            0.5 * np.random.randn(len(X))      # 노이즈
        )

        # 0-100 범위로 클리핑
        satisfaction = np.clip(satisfaction, 0, 100)

        return satisfaction

    def prepare_data(self, X, y, test_size=0.2):
        """
        데이터 전처리 및 분할

        Parameters:
        X (DataFrame): 입력 특성
        y (array): 타겟 변수
        test_size (float): 테스트 크기 비율

        Returns:
        tuple: 전처리된 데이터
        """
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        print(f"✅ 데이터 전처리 완료")
        print(f"   - 학습 데이터: {self.X_train.shape}")
        print(f"   - 테스트 데이터: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, model_type='random_forest'):
        """
        설명 가능한 모델 학습

        Parameters:
        model_type (str): 모델 타입

        Returns:
        object: 학습된 모델
        """
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        # 모델 학습
        self.model.fit(self.X_train, self.y_train)

        # 성능 평가
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)

        print(f"✅ {model_type} 모델 학습 완료")
        print(f"   - 학습 R²: {train_r2:.4f}")
        print(f"   - 테스트 R²: {test_r2:.4f}")

        return self.model

    def setup_shap_explainer(self, explainer_type='tree'):
        """
        SHAP 설명기 설정

        Parameters:
        explainer_type (str): 설명기 타입

        Returns:
        object: SHAP 설명기
        """
        if explainer_type == 'tree':
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            # 커널 설명기는 샘플링된 배경 데이터 사용
            background = shap.sample(self.X_train, 100)
            self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
        else:
            raise ValueError(f"지원하지 않는 설명기 타입: {explainer_type}")

        print(f"✅ SHAP {explainer_type} 설명기 설정 완료")

        return self.shap_explainer

    def generate_shap_explanations(self, X_explain=None, max_display=10):
        """
        SHAP 설명 생성

        Parameters:
        X_explain (DataFrame): 설명할 데이터 (None이면 테스트 데이터 사용)
        max_display (int): 표시할 최대 특성 수

        Returns:
        array: SHAP 값
        """
        if X_explain is None:
            X_explain = self.X_test.head(100)  # 처음 100개 샘플 사용

        # SHAP 값 계산
        shap_values = self.shap_explainer.shap_values(X_explain)

        print(f"✅ SHAP 값 계산 완료: {shap_values.shape}")

        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_explain,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot - 특성 중요도와 영향 방향')
        plt.tight_layout()
        plt.savefig('c:/practice/chap/chapter05/outputs/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_explain,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance - 특성별 평균 절대 기여도')
        plt.tight_layout()
        plt.savefig('c:/practice/chap/chapter05/outputs/shap_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Waterfall Plot (첫 번째 샘플)
        if len(X_explain) > 0:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.shap_explainer.expected_value,
                    data=X_explain.iloc[0],
                    feature_names=self.feature_names
                ),
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Waterfall Plot - 개별 예측 설명')
            plt.tight_layout()
            plt.savefig('c:/practice/chap/chapter05/outputs/shap_waterfall.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 4. Dependence Plot (상위 2개 특성)
        important_features = np.argsort(np.abs(shap_values).mean(0))[-2:]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for i, feature_idx in enumerate(important_features):
            plt.sca(axes[i])
            shap.dependence_plot(
                feature_idx, shap_values, X_explain,
                feature_names=self.feature_names,
                show=False
            )

        plt.tight_layout()
        plt.savefig('c:/practice/chap/chapter05/outputs/shap_dependence.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 SHAP 시각화 완료:")
        print("   - Summary Plot: practice/chapter05/outputs/shap_summary.png")
        print("   - Feature Importance: practice/chapter05/outputs/shap_importance.png")
        print("   - Waterfall Plot: practice/chapter05/outputs/shap_waterfall.png")
        print("   - Dependence Plot: practice/chapter05/outputs/shap_dependence.png")

        return shap_values

    def setup_lime_explainer(self):
        """
        LIME 설명기 설정

        Returns:
        object: LIME 설명기
        """
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            mode='regression',
            discretize_continuous=True,
            random_state=self.random_state
        )

        print("✅ LIME 설명기 설정 완료")

        return self.lime_explainer

    def generate_lime_explanations(self, instance_idx=0, num_features=10):
        """
        LIME 설명 생성

        Parameters:
        instance_idx (int): 설명할 인스턴스 인덱스
        num_features (int): 표시할 특성 수

        Returns:
        object: LIME 설명 객체
        """
        # 설명할 인스턴스 선택
        instance = self.X_test.iloc[instance_idx]
        actual_value = self.y_test[instance_idx]
        predicted_value = self.model.predict([instance.values])[0]

        # LIME 설명 생성
        explanation = self.lime_explainer.explain_instance(
            instance.values,
            self.model.predict,
            num_features=num_features,
            num_samples=1000
        )

        print(f"✅ LIME 설명 생성 완료 (인스턴스 {instance_idx})")
        print(f"   - 실제 값: {actual_value:.2f}")
        print(f"   - 예측 값: {predicted_value:.2f}")

        # 설명 결과 출력
        print(f"\n🎯 LIME 특성 기여도 (상위 {num_features}개):")
        print("-" * 50)

        feature_importance = explanation.as_list()
        for feature, importance in feature_importance:
            direction = "증가" if importance > 0 else "감소"
            print(f"  {feature}: {importance:+.3f} ({direction})")

        # HTML 설명 저장
        explanation.save_to_file('c:/practice/chap/chapter05/outputs/lime_explanation.html')

        # 시각화 생성
        fig = explanation.as_pyplot_figure()
        fig.suptitle(f'LIME 설명 - 인스턴스 {instance_idx}')
        plt.tight_layout()
        plt.savefig('c:/practice/chap/chapter05/outputs/lime_explanation.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 LIME 시각화 완료:")
        print("   - HTML 설명: practice/chapter05/outputs/lime_explanation.html")
        print("   - PNG 이미지: practice/chapter05/outputs/lime_explanation.png")

        return explanation

    def compare_explanations(self, shap_values, lime_explanation, instance_idx=0):
        """
        SHAP과 LIME 설명 비교

        Parameters:
        shap_values (array): SHAP 값
        lime_explanation (object): LIME 설명 객체
        instance_idx (int): 비교할 인스턴스 인덱스

        Returns:
        DataFrame: 비교 결과
        """
        # SHAP 값 추출 (해당 인스턴스)
        shap_instance = shap_values[instance_idx]

        # LIME 값 추출
        lime_dict = dict(lime_explanation.as_list())

        # 비교 데이터프레임 생성
        comparison_data = []

        for i, feature_name in enumerate(self.feature_names):
            shap_value = shap_instance[i]

            # LIME 값 찾기 (특성명이 약간 다를 수 있음)
            lime_value = 0
            for lime_feature, lime_val in lime_dict.items():
                if feature_name in lime_feature or lime_feature in feature_name:
                    lime_value = lime_val
                    break

            comparison_data.append({
                '특성': feature_name,
                'SHAP': shap_value,
                'LIME': lime_value,
                '차이': abs(shap_value - lime_value),
                '방향일치': (shap_value > 0) == (lime_value > 0)
            })

        comparison_df = pd.DataFrame(comparison_data)

        print(f"\n📊 SHAP vs LIME 설명 비교 (인스턴스 {instance_idx}):")
        print("="*60)
        print(comparison_df.to_string(index=False, float_format='%.3f'))

        # 상관관계 계산
        correlation = np.corrcoef(comparison_df['SHAP'], comparison_df['LIME'])[0, 1]
        agreement_rate = comparison_df['방향일치'].mean() * 100

        print(f"\n📈 일치도 분석:")
        print(f"   - 상관계수: {correlation:.3f}")
        print(f"   - 방향 일치율: {agreement_rate:.1f}%")

        # 비교 시각화
        plt.figure(figsize=(12, 5))

        # 서브플롯 1: 값 비교
        plt.subplot(1, 2, 1)
        x_pos = np.arange(len(self.feature_names))
        width = 0.35

        plt.bar(x_pos - width/2, comparison_df['SHAP'], width, label='SHAP', alpha=0.8)
        plt.bar(x_pos + width/2, comparison_df['LIME'], width, label='LIME', alpha=0.8)

        plt.xlabel('특성')
        plt.ylabel('기여도')
        plt.title('SHAP vs LIME 특성 기여도 비교')
        plt.xticks(x_pos, self.feature_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 서브플롯 2: 산점도
        plt.subplot(1, 2, 2)
        plt.scatter(comparison_df['SHAP'], comparison_df['LIME'], alpha=0.7, s=50)
        plt.xlabel('SHAP 값')
        plt.ylabel('LIME 값')
        plt.title(f'상관관계 (r = {correlation:.3f})')

        # 대각선 그리기
        min_val = min(comparison_df['SHAP'].min(), comparison_df['LIME'].min())
        max_val = max(comparison_df['SHAP'].max(), comparison_df['LIME'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('c:/practice/chap/chapter05/outputs/shap_lime_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📈 비교 시각화 저장: practice/chapter05/outputs/shap_lime_comparison.png")

        return comparison_df

def main():
    """메인 실행 함수"""
    print("🚀 설명가능한 AI 구현 시작")
    print("="*60)

    # 설명가능성 구현 객체 생성
    explainer = ExplainabilityImplementation()

    # 1. 데이터 생성
    print("\n📋 1단계: 정책 설명용 데이터 생성")
    X, y, feature_names = explainer.generate_policy_explanation_data(n_samples=1000)

    # 2. 데이터 전처리
    print("\n⚙️ 2단계: 데이터 전처리")
    explainer.prepare_data(X, y, test_size=0.2)

    # 3. 모델 학습
    print("\n🤖 3단계: Random Forest 모델 학습")
    explainer.train_model(model_type='random_forest')

    # 4. SHAP 설명 생성
    print("\n🔍 4단계: SHAP 설명 생성")
    explainer.setup_shap_explainer(explainer_type='tree')
    shap_values = explainer.generate_shap_explanations(max_display=10)

    # 5. LIME 설명 생성
    print("\n🔍 5단계: LIME 설명 생성")
    explainer.setup_lime_explainer()
    lime_explanation = explainer.generate_lime_explanations(instance_idx=0, num_features=10)

    # 6. SHAP과 LIME 비교
    print("\n📊 6단계: SHAP vs LIME 설명 비교")
    comparison_results = explainer.compare_explanations(
        shap_values, lime_explanation, instance_idx=0
    )

    # 7. 추가 분석: 여러 인스턴스에 대한 LIME 설명
    print("\n🔍 7단계: 추가 LIME 분석")
    for i in [1, 2, 3]:
        print(f"\n   📌 인스턴스 {i} LIME 분석:")
        explainer.generate_lime_explanations(instance_idx=i, num_features=5)

    # 8. 최종 요약
    print("\n" + "="*60)
    print("🎯 설명가능한 AI 구현 완료!")
    print("="*60)

    # 모델 성능 요약
    test_pred = explainer.model.predict(explainer.X_test)
    test_r2 = r2_score(explainer.y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(explainer.y_test, test_pred))

    print(f"📊 모델 성능:")
    print(f"   - R²: {test_r2:.4f}")
    print(f"   - RMSE: {test_rmse:.4f}")

    print(f"\n🔍 설명가능성 분석:")
    print(f"   - SHAP 분석 완료: {shap_values.shape[0]}개 인스턴스")
    print(f"   - LIME 분석 완료: 4개 인스턴스")
    print(f"   - SHAP-LIME 상관계수: {np.corrcoef(comparison_results['SHAP'], comparison_results['LIME'])[0,1]:.3f}")

    print(f"\n📁 생성된 파일:")
    print("   - practice/chapter05/outputs/shap_summary.png")
    print("   - practice/chapter05/outputs/shap_importance.png")
    print("   - practice/chapter05/outputs/shap_waterfall.png")
    print("   - practice/chapter05/outputs/shap_dependence.png")
    print("   - practice/chapter05/outputs/lime_explanation.html")
    print("   - practice/chapter05/outputs/lime_explanation.png")
    print("   - practice/chapter05/outputs/shap_lime_comparison.png")

    print("\n✅ 모든 설명가능성 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()