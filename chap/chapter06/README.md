# 제6장: 그래프 이론과 정책 네트워크 분석

본 디렉토리는 제6장 "그래프 이론과 정책 네트워크 분석"의 실습 코드, 데이터, 결과물을 포함합니다.

## 📁 디렉토리 구조

```
practice/chapter06/
├── code/                    # 실습 코드
├── data/                    # 데이터 파일
├── outputs/                 # 시각화 결과물
└── README.md               # 이 파일
```

## 🔧 코드 파일 설명

### 1. `06-government-network.py`
**한국 정부 부처 간 협업 네트워크 생성 및 기본 분석**

- 18개 주요 정부 부처 네트워크 모델링
- 177개 협업 프로젝트 데이터 기반 실제 관계 반영
- 네트워크 기본 속성 분석 (밀도, 연결성, 클러스터링 등)
- 네트워크 시각화 및 데이터 내보내기

**주요 기능:**
- `create_government_network()`: 정부 부처 네트워크 생성
- `analyze_network_properties()`: 네트워크 속성 분석
- `visualize_network()`: 네트워크 시각화
- `export_network_data()`: 다양한 형식으로 데이터 내보내기

### 2. `06-centrality-analysis.py`
**네트워크 중심성 분석과 정책 영향력 측정**

- 다양한 중심성 지표 계산 (연결, 근접, 매개, 고유벡터 중심성)
- 종합 정책 영향력 점수 산출
- 중심성 지표 간 상관관계 분석
- 레이더 차트 및 시각화

**주요 기능:**
- `calculate_all_centralities()`: 모든 중심성 지표 계산
- `calculate_influence_score()`: 종합 영향력 점수 계산
- `analyze_centrality_correlations()`: 상관관계 분석
- `create_radar_chart()`: 다차원 중심성 레이더 차트

### 3. `06-community-detection.py`
**커뮤니티 탐지와 정책 연합 분석**

- Louvain, Label Propagation, Girvan-Newman 알고리즘
- 기능적 분류 기반 커뮤니티
- 정책 주제 식별 및 연합 특성 분석
- 커뮤니티 구조 시각화

**주요 기능:**
- `detect_policy_coalitions()`: 다양한 방법으로 커뮤니티 탐지
- `analyze_community_characteristics()`: 커뮤니티 특성 분석
- `identify_policy_themes()`: 정책 주제 식별
- `create_community_network_layout()`: 커뮤니티 구조 시각화

## 📊 데이터 파일

### 입력 데이터
- 한국 정부 18개 주요 부처 정보
- 177개 부처 간 협업 프로젝트 데이터
- 부처별 예산 규모, 인력, 설립연도 등 속성 정보

### 출력 데이터
- `government_network.graphml`: GraphML 형식 네트워크
- `government_network.json`: JSON 형식 네트워크
- `government_nodes.csv`: 노드(부처) 정보
- `government_edges.csv`: 엣지(협업) 정보
- `centrality_analysis.csv`: 중심성 분석 결과
- `centrality_rankings.xlsx`: 중심성별 순위
- `community_summary.csv`: 커뮤니티 탐지 요약
- `community_detailed.xlsx`: 상세 커뮤니티 분석

## 🎨 시각화 결과

### 네트워크 시각화
- `government_network.png`: 전체 정부 부처 네트워크
- `centrality_analysis.png`: 중심성 분석 시각화
- `centrality_radar.png`: 다차원 중심성 레이더 차트
- `community_analysis.png`: 커뮤니티 분석 결과
- `community_layout.png`: 정책 연합 구조 시각화

## 🚀 실행 방법

### 가상환경 설정 (권장)
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
./venv/Scripts/activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 필수 라이브러리 설치
```bash
# 가상환경 활성화 후 실행
pip install networkx pandas matplotlib seaborn numpy scipy python-louvain
```

### 선택적 라이브러리 (고급 기능용)
```bash
pip install torch torch-geometric  # GNN 분석 (향후 확장)
```

**주의사항**:
- python-louvain 패키지는 Louvain 알고리즘의 최적 구현을 제공합니다
- 패키지가 없을 경우 NetworkX의 대체 알고리즘(Greedy Modularity)이 사용됩니다

### 코드 실행 순서
1. **기본 네트워크 생성**
   ```bash
   python 06-government-network.py
   ```

2. **중심성 분석**
   ```bash
   python 06-centrality-analysis.py
   ```

3. **커뮤니티 분석**
   ```bash
   python 06-community-detection.py
   ```

## 📈 주요 분석 결과

### 네트워크 기본 통계
- 노드 수: 18개 부처
- 엣지 수: 26개 협업 관계
- 총 협업 프로젝트: 177개
- 네트워크 밀도: ~0.085

### 중심성 분석 발견사항
- **과학기술정보통신부**: 최고 연결 중심성 (디지털 전환 중심 역할)
- **기획재정부**: 최고 매개 중심성 (정책 조정 기능)
- **행정안전부**: 높은 근접 중심성 (행정 접근성)

### 정책 연합 구조
1. **디지털 혁신 연합**: 과기부, 행안부, 교육부
2. **경제 정책 연합**: 기재부, 산업부, 중기부
3. **사회 정책 연합**: 복지부, 고용부, 여가부
4. **인프라 환경 연합**: 국토부, 환경부, 해수부, 농식품부
5. **안보 외교 연합**: 외교부, 통일부, 국방부, 법무부

## 🔍 분석 방법론

### 사용된 알고리즘
- **중심성 측정**: Freeman의 4가지 중심성 + Katz 중심성
- **커뮤니티 탐지**: Louvain, Label Propagation, Girvan-Newman
- **시각화**: Spring layout, Force-directed layout
- **통계 분석**: Pearson 상관계수, 모듈러리티

### 데이터 검증
- 실제 정부 조직도 기반 네트워크 구조
- 2025년 현재 177개 협업 프로젝트 반영
- 공공데이터 포털 기반 부처 정보 활용

## ⚠️ 주의사항

### 데이터 제한사항
- 일부 협업 관계는 공개 정보 부족으로 추정값 사용
- 비공식적 협력 관계는 반영되지 않음
- 시간적 변화는 단일 시점 스냅샷으로 제한

### 해석상 주의점
- 중심성 높다고 반드시 정책 영향력이 큰 것은 아님
- 커뮤니티 탐지 결과는 알고리즘에 따라 다를 수 있음
- 정책 연합은 구조적 패턴일 뿐 실제 연합과 다를 수 있음

## 📚 참고 문헌

1. Freeman, L. C. (1978). Centrality in social networks conceptual clarification.
2. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks.
3. Newman, M. E. J. (2006). Modularity and community structure in networks.
4. 한국 정부 조직도 (2025년 기준)
5. 정부 부처 간 협업 프로젝트 현황 (2025년)

## 📞 문의

기술적 문제나 분석 관련 질문이 있으시면 코드 내 주석을 참고하거나 관련 문서를 확인하시기 바랍니다.

---

**생성일**: 2025년 9월 27일
**버전**: 1.0
**호환성**: Python 3.7+, NetworkX 2.5+