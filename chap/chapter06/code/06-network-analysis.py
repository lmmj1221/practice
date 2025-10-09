"""
제6장: 그래프 이론과 정책 네트워크 분석
06-network-analysis.py - 네트워크 속성 상세 분석
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def create_sample_policy_network():
    """정책 네트워크 예제 생성"""
    G = nx.DiGraph()
    
    # 정책 행위자 노드 추가
    actors = {
        'GOV': {'name': '정부', 'type': 'government', 'power': 10},
        'PARL': {'name': '국회', 'type': 'government', 'power': 9},
        'BIZ1': {'name': '대기업', 'type': 'business', 'power': 8},
        'BIZ2': {'name': '중소기업', 'type': 'business', 'power': 5},
        'NGO1': {'name': '환경단체', 'type': 'ngo', 'power': 6},
        'NGO2': {'name': '시민단체', 'type': 'ngo', 'power': 5},
        'ACAD': {'name': '학계', 'type': 'academic', 'power': 7},
        'MEDIA': {'name': '언론', 'type': 'media', 'power': 8},
        'UNION': {'name': '노동조합', 'type': 'union', 'power': 6},
        'LOCAL': {'name': '지방정부', 'type': 'government', 'power': 6}
    }
    
    for node_id, attributes in actors.items():
        G.add_node(node_id, **attributes)
    
    # 정책 영향 관계 추가
    influences = [
        ('GOV', 'PARL', {'weight': 0.8, 'type': 'policy_proposal'}),
        ('PARL', 'GOV', {'weight': 0.7, 'type': 'legislation'}),
        ('BIZ1', 'GOV', {'weight': 0.6, 'type': 'lobbying'}),
        ('BIZ1', 'PARL', {'weight': 0.5, 'type': 'lobbying'}),
        ('NGO1', 'MEDIA', {'weight': 0.7, 'type': 'advocacy'}),
        ('NGO2', 'MEDIA', {'weight': 0.6, 'type': 'advocacy'}),
        ('MEDIA', 'GOV', {'weight': 0.5, 'type': 'public_pressure'}),
        ('MEDIA', 'PARL', {'weight': 0.5, 'type': 'public_pressure'}),
        ('ACAD', 'GOV', {'weight': 0.6, 'type': 'expertise'}),
        ('ACAD', 'NGO1', {'weight': 0.5, 'type': 'research'}),
        ('UNION', 'PARL', {'weight': 0.5, 'type': 'representation'}),
        ('UNION', 'BIZ1', {'weight': 0.4, 'type': 'negotiation'}),
        ('LOCAL', 'GOV', {'weight': 0.6, 'type': 'coordination'}),
        ('GOV', 'LOCAL', {'weight': 0.7, 'type': 'policy_implementation'}),
        ('BIZ2', 'LOCAL', {'weight': 0.4, 'type': 'local_business'})
    ]
    
    G.add_edges_from(influences)
    
    return G

def analyze_network_metrics(G):
    """네트워크 메트릭 상세 분석"""
    
    metrics = {}
    
    # 1. 기본 메트릭
    metrics['basic'] = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'reciprocity': nx.reciprocity(G) if G.is_directed() else None
    }
    
    # 2. 차수 분포
    degrees = dict(G.degree())
    in_degrees = dict(G.in_degree()) if G.is_directed() else degrees
    out_degrees = dict(G.out_degree()) if G.is_directed() else degrees
    
    metrics['degree'] = {
        'mean_degree': np.mean(list(degrees.values())),
        'std_degree': np.std(list(degrees.values())),
        'max_degree': max(degrees.values()),
        'min_degree': min(degrees.values())
    }
    
    # 3. 연결성
    if G.is_directed():
        metrics['connectivity'] = {
            'strongly_connected': nx.is_strongly_connected(G),
            'weakly_connected': nx.is_weakly_connected(G),
            'n_strong_components': nx.number_strongly_connected_components(G),
            'n_weak_components': nx.number_weakly_connected_components(G)
        }
    else:
        metrics['connectivity'] = {
            'is_connected': nx.is_connected(G),
            'n_components': nx.number_connected_components(G)
        }
    
    # 4. 경로 통계
    G_undirected = G.to_undirected() if G.is_directed() else G
    if nx.is_connected(G_undirected):
        metrics['paths'] = {
            'avg_shortest_path': nx.average_shortest_path_length(G_undirected),
            'diameter': nx.diameter(G_undirected),
            'radius': nx.radius(G_undirected)
        }
    
    # 5. 클러스터링
    metrics['clustering'] = {
        'avg_clustering': nx.average_clustering(G_undirected),
        'transitivity': nx.transitivity(G_undirected)
    }
    
    # 6. 중심화 (Centralization)
    degree_cent = nx.degree_centrality(G)
    max_degree_cent = max(degree_cent.values())
    sum_diff = sum(max_degree_cent - cent for cent in degree_cent.values())
    max_sum_diff = (len(G) - 1) * (len(G) - 2)
    
    metrics['centralization'] = {
        'degree_centralization': sum_diff / max_sum_diff if max_sum_diff > 0 else 0
    }
    
    return metrics

def plot_degree_distribution(G):
    """차수 분포 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # In-degree distribution
    in_degrees = [d for n, d in G.in_degree()]
    axes[0].hist(in_degrees, bins=range(max(in_degrees) + 2), 
                 alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('In-Degree Distribution')
    axes[0].set_xlabel('In-Degree')
    axes[0].set_ylabel('Frequency')
    
    # Out-degree distribution
    out_degrees = [d for n, d in G.out_degree()]
    axes[1].hist(out_degrees, bins=range(max(out_degrees) + 2), 
                 alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Out-Degree Distribution')
    axes[1].set_xlabel('Out-Degree')
    axes[1].set_ylabel('Frequency')
    
    # Total degree distribution
    total_degrees = [d for n, d in G.degree()]
    axes[2].hist(total_degrees, bins=range(max(total_degrees) + 2), 
                 alpha=0.7, color='red', edgecolor='black')
    axes[2].set_title('Total Degree Distribution')
    axes[2].set_xlabel('Degree')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../outputs/degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_path_statistics(G):
    """경로 통계 분석"""
    G_undirected = G.to_undirected()
    
    if not nx.is_connected(G_undirected):
        print("네트워크가 연결되어 있지 않습니다.")
        return None
    
    # 모든 최단 경로 길이 계산
    all_paths = dict(nx.all_pairs_shortest_path_length(G_undirected))
    
    # 경로 길이 분포
    path_lengths = []
    for source in all_paths:
        for target, length in all_paths[source].items():
            if source != target:
                path_lengths.append(length)
    
    # 통계 계산
    stats = {
        'mean_path_length': np.mean(path_lengths),
        'median_path_length': np.median(path_lengths),
        'max_path_length': max(path_lengths),
        'min_path_length': min(path_lengths),
        'std_path_length': np.std(path_lengths)
    }
    
    # 경로 길이 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(path_lengths, bins=range(1, max(path_lengths) + 2), 
             alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(stats['mean_path_length'], color='red', linestyle='--', 
                label=f"Mean: {stats['mean_path_length']:.2f}")
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Shortest Path Lengths')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('../outputs/path_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats

def analyze_clustering_patterns(G):
    """클러스터링 패턴 분석"""
    G_undirected = G.to_undirected()
    
    # 노드별 클러스터링 계수
    clustering_coeffs = nx.clustering(G_undirected)
    
    # 차수별 평균 클러스터링
    degree_clustering = {}
    for node in G_undirected.nodes():
        degree = G_undirected.degree(node)
        if degree not in degree_clustering:
            degree_clustering[degree] = []
        degree_clustering[degree].append(clustering_coeffs[node])
    
    avg_clustering_by_degree = {
        k: np.mean(v) for k, v in degree_clustering.items()
    }
    
    # 시각화
    plt.figure(figsize=(10, 6))
    degrees = list(avg_clustering_by_degree.keys())
    clusterings = list(avg_clustering_by_degree.values())
    
    plt.scatter(degrees, clusterings, s=100, alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Clustering Coefficient vs Degree')
    plt.grid(alpha=0.3)
    
    # 추세선 추가
    if len(degrees) > 1:
        z = np.polyfit(degrees, clusterings, 1)
        p = np.poly1d(z)
        plt.plot(degrees, p(degrees), "r--", alpha=0.5, 
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        plt.legend()
    
    plt.savefig('../outputs/clustering_vs_degree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return clustering_coeffs, avg_clustering_by_degree

def create_network_summary_table(G, metrics):
    """네트워크 요약 테이블 생성"""
    
    summary_data = []
    
    # 기본 메트릭
    for key, value in metrics['basic'].items():
        if value is not None:
            summary_data.append({
                'Category': 'Basic',
                'Metric': key.replace('_', ' ').title(),
                'Value': f"{value:.3f}" if isinstance(value, float) else str(value)
            })
    
    # 차수 통계
    for key, value in metrics['degree'].items():
        summary_data.append({
            'Category': 'Degree',
            'Metric': key.replace('_', ' ').title(),
            'Value': f"{value:.3f}" if isinstance(value, float) else str(value)
        })
    
    # 연결성
    for key, value in metrics['connectivity'].items():
        summary_data.append({
            'Category': 'Connectivity',
            'Metric': key.replace('_', ' ').title(),
            'Value': str(value)
        })
    
    # 경로 통계
    if 'paths' in metrics:
        for key, value in metrics['paths'].items():
            summary_data.append({
                'Category': 'Paths',
                'Metric': key.replace('_', ' ').title(),
                'Value': f"{value:.3f}" if isinstance(value, float) else str(value)
            })
    
    # 클러스터링
    for key, value in metrics['clustering'].items():
        summary_data.append({
            'Category': 'Clustering',
            'Metric': key.replace('_', ' ').title(),
            'Value': f"{value:.3f}" if isinstance(value, float) else str(value)
        })
    
    # DataFrame 생성
    df_summary = pd.DataFrame(summary_data)
    
    return df_summary

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("정책 네트워크 상세 분석")
    print("=" * 60)
    
    # 네트워크 생성
    G = create_sample_policy_network()
    
    # 메트릭 분석
    metrics = analyze_network_metrics(G)
    
    # 요약 테이블 생성 및 출력
    summary_table = create_network_summary_table(G, metrics)
    print("\n[네트워크 메트릭 요약]")
    print(summary_table.to_string(index=False))
    
    # 차수 분포 시각화
    print("\n차수 분포 분석 중...")
    plot_degree_distribution(G)
    
    # 경로 통계 분석
    print("\n경로 통계 분석 중...")
    path_stats = analyze_path_statistics(G)
    if path_stats:
        print("\n[경로 통계]")
        for key, value in path_stats.items():
            print(f"  {key}: {value:.3f}")
    
    # 클러스터링 패턴 분석
    print("\n클러스터링 패턴 분석 중...")
    clustering_coeffs, avg_clustering = analyze_clustering_patterns(G)
    
    # 결과 저장
    summary_table.to_csv('../outputs/network_metrics_summary.csv', index=False)
    print("\n분석 결과가 저장되었습니다.")
    
    return G, metrics

if __name__ == "__main__":
    network, metrics = main()