# Example 1.2: Dynamic Public Policy-Cycle Data Processing Pipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
current_dir = Path(__file__).parent
output_dir = current_dir.parent / 'outputs'
output_dir.mkdir(exist_ok=True)

class DPPCPipeline:
    """Dynamic Public Policy-Cycle Pipeline Implementation"""
    
    def __init__(self):
        self.stages = {
            1: 'Agenda Setting',
            2: 'Policy Formulation', 
            3: 'Decision Making',
            4: 'Implementation',
            5: 'Evaluation'
        }
        self.feedback_loops = []
        self.stage_data = {}
        
    def process_stage(self, stage_num, data, ai_methods):
        """Process each policy stage and apply AI methods"""
        stage_name = self.stages[stage_num]
        print(f"\n[Stage {stage_num}] {stage_name}")
        print(f"Processing {len(data)} data points")
        print(f"AI Methods: {', '.join(ai_methods)}")
        
        # Processing logic for each stage
        if stage_num == 1:  # Agenda Setting
            result = self._agenda_setting(data, ai_methods)
        elif stage_num == 2:  # Policy Formulation
            result = self._policy_formulation(data, ai_methods)
        elif stage_num == 3:  # Decision Making
            result = self._decision_making(data, ai_methods)
        elif stage_num == 4:  # Implementation
            result = self._implementation(data, ai_methods)
        else:  # Evaluation
            result = self._evaluation(data, ai_methods)
            
        self.stage_data[stage_num] = result
        return result
    
    def _agenda_setting(self, data, ai_methods):
        """Process agenda setting stage"""
        # Calculate citizen proposal priority scores (simulation)
        priority_scores = np.random.beta(2, 5, len(data)) * 100
        return {'priorities': priority_scores, 'selected_issues': len(priority_scores[priority_scores > 70])}
    
    def _policy_formulation(self, data, ai_methods):
        """Process policy formulation stage"""
        # 정책 대안 생성 및 시뮬레이션 점수
        alternatives = np.random.poisson(3, 5) + 1
        simulation_scores = np.random.normal(75, 10, len(alternatives))
        return {'alternatives': alternatives, 'scores': simulation_scores}
    
    def _decision_making(self, data, ai_methods):
        """의사결정 단계 처리"""
        # 다기준 의사결정 점수 계산
        criteria = ['Effectiveness', 'Efficiency', 'Equity', 'Feasibility']
        scores = np.random.uniform(60, 95, (3, len(criteria)))
        return {'criteria': criteria, 'decision_matrix': scores}
    
    def _implementation(self, data, ai_methods):
        """집행 단계 처리"""
        # 집행 성과 추적
        days = 30
        performance = np.cumsum(np.random.normal(2, 0.5, days))
        return {'days': days, 'cumulative_performance': performance}
    
    def _evaluation(self, data, ai_methods):
        """평가 단계 처리"""
        # 정책 효과 측정
        baseline = 50
        treatment_effect = np.random.normal(15, 3)
        return {'baseline': baseline, 'effect': treatment_effect, 'total': baseline + treatment_effect}
    
    def create_feedback_loop(self, from_stage, to_stage, data):
        """피드백 루프 생성"""
        feedback = {
            'from': self.stages[from_stage],
            'to': self.stages[to_stage],
            'timestamp': datetime.now(),
            'data': data
        }
        self.feedback_loops.append(feedback)
        print(f"\nFeedback Loop Created: {self.stages[from_stage]} → {self.stages[to_stage]}")
        return feedback
    
    def visualize_pipeline(self):
        """파이프라인 시각화"""
        from pathlib import Path
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent / 'outputs'
        output_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dynamic Public Policy-Cycle Pipeline Visualization', fontsize=14, fontweight='bold')
        
        # 1. 단계별 진행 상황
        ax1 = axes[0, 0]
        stages = list(self.stages.values())
        progress = [85, 75, 90, 60, 95]  # 시뮬레이션 진행률
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(stages)))
        bars = ax1.barh(stages, progress, color=colors)
        ax1.set_xlabel('Progress (%)')
        ax1.set_title('Stage Progress Status')
        ax1.set_xlim(0, 100)
        for i, (bar, val) in enumerate(zip(bars, progress)):
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val}%', va='center')
        
        # 2. 피드백 루프 네트워크
        ax2 = axes[0, 1]
        feedback_matrix = np.array([
            [0, 2, 1, 0, 3],
            [2, 0, 3, 1, 0],
            [1, 3, 0, 2, 1],
            [0, 1, 2, 0, 4],
            [3, 0, 1, 4, 0]
        ])
        im = ax2.imshow(feedback_matrix, cmap='YlOrRd')
        ax2.set_xticks(range(5))
        ax2.set_yticks(range(5))
        ax2.set_xticklabels(['AS', 'PF', 'DM', 'IM', 'EV'])
        ax2.set_yticklabels(['AS', 'PF', 'DM', 'IM', 'EV'])
        ax2.set_title('Feedback Loop Intensity')
        plt.colorbar(im, ax=ax2)
        
        # 3. AI 기법 활용도
        ax3 = axes[1, 0]
        ai_methods = ['NLP', 'ML', 'Optimization', 'Simulation', 'Causal AI']
        usage = [78, 85, 72, 90, 65]
        ax3.pie(usage, labels=ai_methods, autopct='%1.1f%%', startangle=90)
        ax3.set_title('AI Methods Utilization')
        
        # 4. 시간별 처리량
        ax4 = axes[1, 1]
        hours = range(24)
        throughput = np.random.poisson(50, 24) + np.sin(np.linspace(0, 2*np.pi, 24)) * 10 + 50
        ax4.plot(hours, throughput, 'b-', linewidth=2)
        ax4.fill_between(hours, throughput, alpha=0.3)
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Requests Processed')
        ax4.set_title('24-Hour Processing Throughput')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dppc_pipeline.png', dpi=150, bbox_inches='tight')
        plt.show()

# DPPC 파이프라인 실행
print("=== Dynamic Public Policy-Cycle Pipeline Demo ===")
pipeline = DPPCPipeline()

# 각 단계 실행
sample_data = np.random.randn(100)
pipeline.process_stage(1, sample_data, ['Text Mining', 'Sentiment Analysis', 'Topic Modeling'])
pipeline.process_stage(2, sample_data, ['Predictive Modeling', 'Simulation', 'Optimization'])
pipeline.process_stage(3, sample_data, ['MCDA', 'XAI', 'Consensus Algorithm'])
pipeline.process_stage(4, sample_data, ['Process Mining', 'Anomaly Detection', 'Resource Optimization'])
pipeline.process_stage(5, sample_data, ['Causal Inference', 'Performance Prediction', 'Benchmarking'])

# 피드백 루프 생성
pipeline.create_feedback_loop(5, 1, {'satisfaction_score': 82, 'improvement_areas': ['efficiency', 'transparency']})
pipeline.create_feedback_loop(4, 2, {'execution_bottlenecks': ['resource_shortage', 'coordination_issues']})

# 시각화
pipeline.visualize_pipeline()

print("\n=== Pipeline Summary ===")
print(f"Total Stages Processed: {len(pipeline.stage_data)}")
print(f"Feedback Loops Created: {len(pipeline.feedback_loops)}")
print(f"Average Processing Time: {np.random.uniform(1.2, 2.5):.2f} seconds per stage")