#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
제5장: AI 에이전트 기반 정책 지원 시스템
다중 에이전트 환경에서의 정책 분석 및 의사결정 지원
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

@dataclass
class PolicyRecommendation:
    """정책 권장사항 데이터 클래스"""
    policy_name: str
    priority: float
    confidence: float
    impact_score: float
    rationale: str
    estimated_effect: Dict[str, float]

@dataclass
class AgentDecision:
    """에이전트 의사결정 데이터 클래스"""
    agent_type: str
    decision: str
    confidence: float
    reasoning: List[str]
    data_sources: List[str]

class PolicyAnalysisAgent:
    """정책 분석 전문 에이전트"""

    def __init__(self, agent_id: str, specialization: str):
        """
        정책 분석 에이전트 초기화

        Parameters:
        agent_id (str): 에이전트 식별자
        specialization (str): 전문 분야
        """
        self.agent_id = agent_id
        self.specialization = specialization
        self.model = None
        self.knowledge_base = {}
        self.decision_history = []

    def train_model(self, X: np.ndarray, y: np.ndarray, model_type='random_forest'):
        """
        에이전트 내부 모델 학습

        Parameters:
        X (array): 입력 특성
        y (array): 타겟 변수
        model_type (str): 모델 타입
        """
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X, y)

        # 성능 평가
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)

        print(f"✅ {self.agent_id} 모델 학습 완료 (R²: {r2:.4f})")

    def analyze_policy_scenario(self, scenario_data: Dict[str, float]) -> PolicyRecommendation:
        """
        정책 시나리오 분석

        Parameters:
        scenario_data (dict): 시나리오 데이터

        Returns:
        PolicyRecommendation: 정책 권장사항
        """
        # 시나리오 데이터를 배열로 변환
        features = np.array([list(scenario_data.values())]).reshape(1, -1)

        # 예측 수행
        if self.model is not None:
            prediction = self.model.predict(features)[0]

            # 특성 중요도 기반 신뢰도 계산
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                confidence = np.mean(importance_scores) * 0.8 + 0.2
            else:
                confidence = 0.7
        else:
            prediction = np.random.uniform(50, 80)
            confidence = 0.5

        # 정책 권장사항 생성
        policy_name = f"{self.specialization}_정책_개선안"
        priority = min(prediction / 100, 1.0)
        impact_score = prediction

        # 추론 논리 생성
        rationale = self._generate_rationale(scenario_data, prediction)

        # 예상 효과 계산
        estimated_effect = self._calculate_estimated_effect(scenario_data, prediction)

        recommendation = PolicyRecommendation(
            policy_name=policy_name,
            priority=priority,
            confidence=confidence,
            impact_score=impact_score,
            rationale=rationale,
            estimated_effect=estimated_effect
        )

        return recommendation

    def _generate_rationale(self, scenario_data: Dict[str, float], prediction: float) -> str:
        """추론 논리 생성"""
        key_factors = []

        # 시나리오 데이터에서 주요 요인 식별
        for key, value in scenario_data.items():
            if value > 0.6:  # 높은 값
                key_factors.append(f"{key} 수준이 높음 ({value:.2f})")
            elif value < 0.4:  # 낮은 값
                key_factors.append(f"{key} 수준이 낮음 ({value:.2f})")

        if prediction > 70:
            conclusion = "긍정적인 정책 효과가 예상됨"
        elif prediction > 50:
            conclusion = "중간 수준의 정책 효과가 예상됨"
        else:
            conclusion = "정책 효과 개선이 필요함"

        rationale = f"{self.specialization} 관점에서 {', '.join(key_factors[:2])}를 고려할 때, {conclusion}"

        return rationale

    def _calculate_estimated_effect(self, scenario_data: Dict[str, float], prediction: float) -> Dict[str, float]:
        """예상 효과 계산"""
        base_effect = prediction / 100

        estimated_effect = {
            '경제적_효과': base_effect * 0.8 + np.random.normal(0, 0.1),
            '사회적_효과': base_effect * 0.9 + np.random.normal(0, 0.1),
            '환경적_효과': base_effect * 0.7 + np.random.normal(0, 0.1),
            '정치적_효과': base_effect * 0.6 + np.random.normal(0, 0.1)
        }

        # 0-1 범위로 클리핑
        estimated_effect = {k: np.clip(v, 0, 1) for k, v in estimated_effect.items()}

        return estimated_effect

    def make_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """
        의사결정 수행

        Parameters:
        context (dict): 의사결정 컨텍스트

        Returns:
        AgentDecision: 에이전트 의사결정
        """
        # 결정 옵션들
        options = ['정책_승인', '정책_수정', '정책_거부', '추가_분석_필요']

        # 컨텍스트 기반 점수 계산
        scores = {}
        for option in options:
            score = np.random.uniform(0, 1)

            # 전문분야별 가중치 적용
            if self.specialization == '경제' and 'economic' in option.lower():
                score *= 1.2
            elif self.specialization == '사회' and 'social' in option.lower():
                score *= 1.2
            elif self.specialization == '환경' and 'environmental' in option.lower():
                score *= 1.2

            scores[option] = score

        # 최고 점수 옵션 선택
        best_option = max(scores.keys(), key=lambda x: scores[x])
        confidence = scores[best_option]

        # 추론 과정 생성
        reasoning = [
            f"{self.specialization} 전문성을 바탕으로 분석",
            f"컨텍스트 요소 {len(context)}개 고려",
            f"최적 선택: {best_option} (신뢰도: {confidence:.3f})"
        ]

        # 데이터 소스
        data_sources = ['내부_모델', '전문_지식', '과거_사례']

        decision = AgentDecision(
            agent_type=self.specialization,
            decision=best_option,
            confidence=confidence,
            reasoning=reasoning,
            data_sources=data_sources
        )

        self.decision_history.append(decision)

        return decision

class MultiAgentPolicySystem:
    """다중 에이전트 정책 지원 시스템"""

    def __init__(self):
        """다중 에이전트 시스템 초기화"""
        self.agents = {}
        self.coordination_history = []
        self.consensus_threshold = 0.7

    def create_agents(self, specializations: List[str]):
        """
        전문 분야별 에이전트 생성

        Parameters:
        specializations (list): 전문 분야 목록
        """
        for spec in specializations:
            agent_id = f"agent_{spec}"
            self.agents[agent_id] = PolicyAnalysisAgent(agent_id, spec)

        print(f"✅ {len(specializations)}개 전문 에이전트 생성 완료")
        for spec in specializations:
            print(f"   - {spec} 전문 에이전트")

    def train_all_agents(self, X: np.ndarray, y: np.ndarray):
        """
        모든 에이전트 학습

        Parameters:
        X (array): 입력 특성
        y (array): 타겟 변수
        """
        print("\n🤖 모든 에이전트 학습 시작")

        for agent_id, agent in self.agents.items():
            # 각 에이전트마다 약간 다른 모델 사용
            model_types = ['random_forest', 'gradient_boosting']
            model_type = model_types[len(agent_id) % 2]

            agent.train_model(X, y, model_type)

        print("✅ 모든 에이전트 학습 완료")

    def generate_policy_data(self, n_samples=1000):
        """
        정책 분석용 데이터 생성
        ※ 본 데이터는 교육 목적의 시뮬레이션 데이터입니다

        Parameters:
        n_samples (int): 샘플 수

        Returns:
        tuple: (X, y, feature_names) 특성, 타겟, 특성명
        """
        np.random.seed(42)

        feature_names = [
            '경제성장률', '교육투자율', '인프라수준',
            '사회복지수준', '환경품질지수', '정치안정성'
        ]

        n_features = len(feature_names)

        # 기본 특성 생성 (0-1 정규화된 값)
        X = np.random.uniform(0, 1, (n_samples, n_features))

        # 복잡한 정책 효과 함수
        policy_effectiveness = (
            0.3 * X[:, 0] +                     # 경제성장률
            0.25 * X[:, 1] +                    # 교육투자율
            0.2 * X[:, 2] +                     # 인프라수준
            0.15 * X[:, 3] +                    # 사회복지수준
            0.1 * X[:, 4] +                     # 환경품질지수
            0.05 * X[:, 5] +                    # 정치안정성
            0.1 * X[:, 0] * X[:, 1] +           # 경제-교육 상호작용
            0.05 * X[:, 2] * X[:, 3] +          # 인프라-복지 상호작용
            0.1 * np.random.randn(n_samples)    # 노이즈
        )

        # 0-100 점수로 스케일링
        y = 50 + 30 * policy_effectiveness
        y = np.clip(y, 0, 100)

        # DataFrame 생성
        X_df = pd.DataFrame(X, columns=feature_names)

        print(f"✅ 정책 분석용 데이터 생성 완료")
        print(f"   - 샘플 수: {n_samples}")
        print(f"   - 특성 수: {n_features}")
        print(f"   - 정책효과 범위: [{y.min():.2f}, {y.max():.2f}]")

        return X_df, y, feature_names

    def coordinate_agents(self, scenario: Dict[str, float]) -> List[PolicyRecommendation]:
        """
        에이전트 간 협력 및 조정

        Parameters:
        scenario (dict): 분석할 시나리오

        Returns:
        list: 각 에이전트의 정책 권장사항
        """
        print(f"\n🤝 에이전트 협력 분석 시작")
        print(f"시나리오: {scenario}")

        recommendations = []

        for agent_id, agent in self.agents.items():
            print(f"   🔄 {agent.specialization} 에이전트 분석 중...")

            recommendation = agent.analyze_policy_scenario(scenario)
            recommendations.append(recommendation)

            print(f"      ✅ 완료 - 우선순위: {recommendation.priority:.3f}, "
                  f"신뢰도: {recommendation.confidence:.3f}")

        print("✅ 모든 에이전트 분석 완료")

        return recommendations

    def make_collective_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        집단 의사결정 수행

        Parameters:
        context (dict): 의사결정 컨텍스트

        Returns:
        dict: 집단 의사결정 결과
        """
        print(f"\n🗳️ 집단 의사결정 시작")

        agent_decisions = []
        decision_scores = {}

        # 각 에이전트의 의사결정 수집
        for agent_id, agent in self.agents.items():
            decision = agent.make_decision(context)
            agent_decisions.append(decision)

            # 의사결정별 점수 집계
            if decision.decision not in decision_scores:
                decision_scores[decision.decision] = []
            decision_scores[decision.decision].append(decision.confidence)

        # 가중 평균 계산
        weighted_scores = {}
        for decision, scores in decision_scores.items():
            weighted_scores[decision] = np.mean(scores)

        # 최종 의사결정
        final_decision = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
        consensus_score = weighted_scores[final_decision]

        # 합의 여부 판단
        consensus_reached = consensus_score >= self.consensus_threshold

        result = {
            'final_decision': final_decision,
            'consensus_score': consensus_score,
            'consensus_reached': consensus_reached,
            'agent_decisions': agent_decisions,
            'vote_distribution': decision_scores
        }

        print(f"✅ 집단 의사결정 완료")
        print(f"   - 최종 결정: {final_decision}")
        print(f"   - 합의 점수: {consensus_score:.3f}")
        print(f"   - 합의 달성: {'예' if consensus_reached else '아니오'}")

        self.coordination_history.append(result)

        return result

    def generate_comprehensive_report(self, recommendations: List[PolicyRecommendation],
                                    collective_decision: Dict[str, Any]) -> str:
        """
        종합 분석 보고서 생성

        Parameters:
        recommendations (list): 정책 권장사항들
        collective_decision (dict): 집단 의사결정 결과

        Returns:
        str: 종합 보고서
        """
        report = []
        report.append("="*80)
        report.append("AI 에이전트 기반 정책 분석 종합 보고서")
        report.append("="*80)

        # 개별 에이전트 분석 결과
        report.append("\n1. 개별 에이전트 분석 결과")
        report.append("-" * 50)

        for i, rec in enumerate(recommendations, 1):
            report.append(f"\n[에이전트 {i}: {rec.policy_name}]")
            report.append(f"  우선순위: {rec.priority:.3f}")
            report.append(f"  신뢰도: {rec.confidence:.3f}")
            report.append(f"  영향도: {rec.impact_score:.2f}")
            report.append(f"  추론: {rec.rationale}")

            report.append("  예상 효과:")
            for effect, value in rec.estimated_effect.items():
                report.append(f"    - {effect}: {value:.3f}")

        # 집단 의사결정 결과
        report.append("\n\n2. 집단 의사결정 결과")
        report.append("-" * 50)
        report.append(f"최종 결정: {collective_decision['final_decision']}")
        report.append(f"합의 점수: {collective_decision['consensus_score']:.3f}")
        report.append(f"합의 달성: {'예' if collective_decision['consensus_reached'] else '아니오'}")

        # 투표 분포
        report.append("\n투표 분포:")
        for decision, scores in collective_decision['vote_distribution'].items():
            avg_score = np.mean(scores)
            vote_count = len(scores)
            report.append(f"  - {decision}: {vote_count}표 (평균 신뢰도: {avg_score:.3f})")

        # 종합 권장사항
        report.append("\n\n3. 종합 권장사항")
        report.append("-" * 50)

        # 우선순위 높은 권장사항
        top_recommendation = max(recommendations, key=lambda x: x.priority)
        report.append(f"최우선 정책: {top_recommendation.policy_name}")
        report.append(f"권장 이유: {top_recommendation.rationale}")

        # 신뢰도 높은 권장사항
        most_confident = max(recommendations, key=lambda x: x.confidence)
        report.append(f"\n가장 확실한 정책: {most_confident.policy_name}")
        report.append(f"신뢰도: {most_confident.confidence:.3f}")

        return "\n".join(report)

    def visualize_agent_analysis(self, recommendations: List[PolicyRecommendation],
                               save_path='practice/chapter05/outputs/agent_analysis.png'):
        """
        에이전트 분석 결과 시각화

        Parameters:
        recommendations (list): 정책 권장사항들
        save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 우선순위 비교
        ax1 = axes[0, 0]
        agent_names = [rec.policy_name.split('_')[0] for rec in recommendations]
        priorities = [rec.priority for rec in recommendations]

        bars1 = ax1.bar(agent_names, priorities, alpha=0.7, color='skyblue')
        ax1.set_title('에이전트별 정책 우선순위')
        ax1.set_ylabel('우선순위 점수')
        ax1.tick_params(axis='x', rotation=45)

        # 값 표시
        for bar, priority in zip(bars1, priorities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{priority:.3f}', ha='center', va='bottom')

        # 2. 신뢰도 비교
        ax2 = axes[0, 1]
        confidences = [rec.confidence for rec in recommendations]

        bars2 = ax2.bar(agent_names, confidences, alpha=0.7, color='lightgreen')
        ax2.set_title('에이전트별 신뢰도')
        ax2.set_ylabel('신뢰도')
        ax2.tick_params(axis='x', rotation=45)

        for bar, confidence in zip(bars2, confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{confidence:.3f}', ha='center', va='bottom')

        # 3. 영향도 점수 비교
        ax3 = axes[1, 0]
        impact_scores = [rec.impact_score for rec in recommendations]

        # 바 차트로 각 에이전트별 영향도 표시
        bars3 = ax3.bar(agent_names, impact_scores, alpha=0.7, color='orange')
        ax3.set_title('Agent Impact Scores')
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Impact Score')
        ax3.tick_params(axis='x', rotation=45)

        for bar, score in zip(bars3, impact_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom')

        # 4. 예상 효과 비교 (첫 번째 권장사항 기준)
        ax4 = axes[1, 1]

        if recommendations:
            effects = recommendations[0].estimated_effect
            effect_names = list(effects.keys())
            effect_values = list(effects.values())

            bars4 = ax4.bar(effect_names, effect_values, alpha=0.7, color='purple')
            ax4.set_title('예상 효과 분석 (첫 번째 에이전트)')
            ax4.set_ylabel('효과 점수')
            ax4.tick_params(axis='x', rotation=45)

            for bar, value in zip(bars4, effect_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 에이전트 분석 시각화 저장: {save_path}")

def main():
    """메인 실행 함수"""
    print("🚀 AI 에이전트 기반 정책 지원 시스템 시작")
    print("="*70)

    # 1. 다중 에이전트 시스템 생성
    print("🤖 다중 에이전트 시스템 초기화")
    mas = MultiAgentPolicySystem()

    # 2. 전문 분야별 에이전트 생성
    specializations = ['경제', '사회', '환경', '기술', '정치']
    mas.create_agents(specializations)

    # 3. 학습 데이터 생성
    print("\n📊 정책 분석용 데이터 생성")
    X, y, feature_names = mas.generate_policy_data(n_samples=1200)

    # 4. 모든 에이전트 학습
    mas.train_all_agents(X.values, y)

    # 5. 정책 시나리오 분석
    print("\n📋 정책 시나리오 분석")
    test_scenario = {
        '경제성장률': 0.7,
        '교육투자율': 0.8,
        '인프라수준': 0.6,
        '사회복지수준': 0.5,
        '환경품질지수': 0.4,
        '정치안정성': 0.9
    }

    # 에이전트 협력 분석
    recommendations = mas.coordinate_agents(test_scenario)

    # 6. 집단 의사결정
    print("\n🗳️ 집단 의사결정 수행")
    decision_context = {
        'urgency': 'high',
        'budget_constraint': 'medium',
        'political_feasibility': 'high',
        'public_support': 'medium'
    }

    collective_decision = mas.make_collective_decision(decision_context)

    # 7. 종합 보고서 생성
    print("\n📄 종합 분석 보고서 생성")
    report = mas.generate_comprehensive_report(recommendations, collective_decision)

    # 보고서 저장
    with open('practice/chapter05/outputs/ai_agent_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("📄 보고서 저장: practice/chapter05/outputs/ai_agent_report.txt")

    # 8. 시각화
    print("\n📈 분석 결과 시각화")
    mas.visualize_agent_analysis(recommendations)

    # 9. 추가 시나리오 테스트
    print("\n🔄 추가 시나리오 테스트")
    additional_scenarios = [
        {'경제성장률': 0.3, '교육투자율': 0.4, '인프라수준': 0.3,
         '사회복지수준': 0.2, '환경품질지수': 0.8, '정치안정성': 0.6},
        {'경제성장률': 0.9, '교육투자율': 0.9, '인프라수준': 0.8,
         '사회복지수준': 0.7, '환경품질지수': 0.6, '정치안정성': 0.8}
    ]

    for i, scenario in enumerate(additional_scenarios, 1):
        print(f"\n   📌 시나리오 {i} 분석:")
        scenario_recs = mas.coordinate_agents(scenario)

        # 최고 우선순위 권장사항 출력
        top_rec = max(scenario_recs, key=lambda x: x.priority)
        print(f"      최우선 정책: {top_rec.policy_name}")
        print(f"      우선순위: {top_rec.priority:.3f}")
        print(f"      추론: {top_rec.rationale}")

    # 10. 최종 요약
    print("\n" + "="*70)
    print("🎯 AI 에이전트 기반 정책 지원 시스템 완료!")
    print("="*70)

    # 성능 요약
    avg_confidence = np.mean([rec.confidence for rec in recommendations])
    avg_priority = np.mean([rec.priority for rec in recommendations])

    print(f"📊 시스템 성능:")
    print(f"   - 활성 에이전트: {len(mas.agents)}개")
    print(f"   - 평균 신뢰도: {avg_confidence:.3f}")
    print(f"   - 평균 우선순위: {avg_priority:.3f}")
    print(f"   - 합의 달성률: {collective_decision['consensus_score']:.1%}")

    print(f"\n🔍 에이전트 협력 분석:")
    print(f"   - 시나리오 분석: 3개 완료")
    print(f"   - 정책 권장사항: {len(recommendations)}개 생성")
    print(f"   - 집단 의사결정: {'성공' if collective_decision['consensus_reached'] else '부분 성공'}")

    print(f"\n📁 생성된 파일:")
    print("   - practice/chapter05/outputs/ai_agent_report.txt")
    print("   - practice/chapter05/outputs/agent_analysis.png")

    print("\n✅ 모든 AI 에이전트 분석이 완료되었습니다!")

if __name__ == "__main__":
    main()