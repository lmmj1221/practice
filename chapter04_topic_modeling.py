"""
제4장: LDA와 BERTopic을 활용한 민원 주제 분석
전처리된 민원 텍스트에서 잠재 주제를 추출하고 분석
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json
import os

# 토픽 모델링 라이브러리
try:
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: Gensim not installed. LDA analysis will be limited.")

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("Warning: BERTopic not installed. Using alternative implementation.")

# 시각화 및 분석
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA
from collections import Counter, defaultdict
import seaborn as sns

# 한국어 글꼴 설정 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)

class TopicAnalyzer:
    """토픽 모델링 분석기"""
    
    def __init__(self, num_topics=10):
        self.num_topics = num_topics
        self.lda_model = None
        self.bertopic_model = None
        self.vectorizer = None
        self.dictionary = None
        self.corpus = None
        
        # 분석 결과 저장
        self.results = {
            'lda_topics': {},
            'bertopic_topics': {},
            'topic_comparison': {},
            'keywords_analysis': {},
            'document_topics': []
        }
    
    def preprocess_for_lda(self, texts):
        """LDA를 위한 텍스트 전처리"""
        processed_docs = []
        for text in texts:
            if pd.isna(text) or not text.strip():
                continue
            # 공백으로 분할하여 토큰 리스트 생성
            tokens = text.split()
            # 길이가 2 이상인 토큰만 유지
            tokens = [token for token in tokens if len(token) >= 2]
            if tokens:
                processed_docs.append(tokens)
        
        return processed_docs
    
    def train_lda_model(self, texts):
        """LDA 모델 학습"""
        print("LDA 모델 학습 시작...")
        
        # 텍스트 전처리
        processed_docs = self.preprocess_for_lda(texts)
        
        if not processed_docs:
            print("Error: No valid documents for LDA training")
            return None
        
        if GENSIM_AVAILABLE:
            try:
                # Gensim LDA 사용
                self.dictionary = corpora.Dictionary(processed_docs)
                self.dictionary.filter_extremes(no_below=2, no_above=0.8)
                self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
                
                self.lda_model = LdaModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    num_topics=self.num_topics,
                    random_state=42,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True
                )
                
                print("Gensim LDA 모델 학습 완료")
                return self.lda_model
                
            except Exception as e:
                print(f"Gensim LDA 학습 실패: {e}")
                
        # Fallback: Scikit-learn LDA 사용
        print("Scikit-learn LDA 모델로 대체 학습...")
        try:
            # TF-IDF 벡터화
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                stop_words=None
            )
            
            # 텍스트를 다시 문자열로 변환
            text_strings = [' '.join(doc) for doc in processed_docs]
            tfidf_matrix = self.vectorizer.fit_transform(text_strings)
            
            # LDA 모델 학습
            self.lda_model = SklearnLDA(
                n_components=self.num_topics,
                random_state=42,
                max_iter=10
            )
            self.lda_model.fit(tfidf_matrix)
            
            print("Scikit-learn LDA 모델 학습 완료")
            return self.lda_model
            
        except Exception as e:
            print(f"LDA 모델 학습 실패: {e}")
            return None
    
    def extract_lda_topics(self):
        """LDA 주제 추출"""
        if not self.lda_model:
            return {}
        
        topics = {}
        
        if GENSIM_AVAILABLE and hasattr(self.lda_model, 'print_topics'):
            # Gensim LDA
            for idx, topic in self.lda_model.print_topics():
                # 주제에서 키워드 추출
                keywords = []
                topic_terms = topic.split('+')
                for term in topic_terms:
                    # 확률과 단어 분리
                    parts = term.strip().split('*')
                    if len(parts) == 2:
                        prob = float(parts[0])
                        word = parts[1].strip().replace('"', '')
                        keywords.append({'word': word, 'weight': prob})
                
                topics[f"주제_{idx+1}"] = {
                    'keywords': keywords[:5],  # 상위 5개 키워드
                    'description': f"LDA 주제 {idx+1}"
                }
        else:
            # Scikit-learn LDA
            if self.vectorizer:
                feature_names = self.vectorizer.get_feature_names_out()
                for topic_idx, topic in enumerate(self.lda_model.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    keywords = []
                    for word_idx in top_words_idx[:5]:
                        word = feature_names[word_idx]
                        weight = topic[word_idx]
                        keywords.append({'word': word, 'weight': float(weight)})
                    
                    topics[f"주제_{topic_idx+1}"] = {
                        'keywords': keywords,
                        'description': f"LDA 주제 {topic_idx+1}"
                    }
        
        self.results['lda_topics'] = topics
        return topics
    
    def train_bertopic_model(self, texts):
        """BERTopic 모델 학습"""
        print("BERTopic 모델 학습 시작...")
        
        if not BERTOPIC_AVAILABLE:
            print("BERTopic 대체 구현으로 진행...")
            return self.simple_topic_modeling(texts)
        
        try:
            # BERTopic 모델 초기화
            self.bertopic_model = BERTopic(
                language="korean",
                nr_topics=self.num_topics,
                verbose=True
            )
            
            # 빈 텍스트 제거
            valid_texts = [text for text in texts if text and text.strip()]
            
            if len(valid_texts) < 10:
                print("Warning: Too few documents for BERTopic")
                return self.simple_topic_modeling(texts)
            
            # 모델 학습
            topics, probabilities = self.bertopic_model.fit_transform(valid_texts)
            
            print("BERTopic 모델 학습 완료")
            return topics, probabilities
            
        except Exception as e:
            print(f"BERTopic 학습 실패: {e}")
            return self.simple_topic_modeling(texts)
    
    def simple_topic_modeling(self, texts):
        """간단한 토픽 모델링 대체 구현"""
        print("간단한 토픽 모델링 구현...")
        
        # 텍스트 전처리 및 키워드 추출
        all_words = []
        for text in texts:
            if text and text.strip():
                words = text.split()
                all_words.extend([word for word in words if len(word) >= 2])
        
        # 단어 빈도 계산
        word_freq = Counter(all_words)
        common_words = word_freq.most_common(50)
        
        # 임의 토픽 생성
        topics = []
        words_per_topic = len(common_words) // self.num_topics
        
        for i in range(self.num_topics):
            start_idx = i * words_per_topic
            end_idx = start_idx + words_per_topic
            topic_words = common_words[start_idx:end_idx]
            topics.append(topic_words[:5])  # 상위 5개 단어
        
        return topics, [0] * len(texts)  # 더미 확률
    
    def extract_bertopic_topics(self):
        """BERTopic 주제 추출"""
        if not self.bertopic_model and not hasattr(self, '_simple_topics'):
            return {}
        
        topics = {}
        
        if BERTOPIC_AVAILABLE and self.bertopic_model:
            # BERTopic에서 주제 정보 추출
            topic_info = self.bertopic_model.get_topic_info()
            for idx, row in topic_info.iterrows():
                if row['Topic'] == -1:  # 노이즈 토픽 제외
                    continue
                
                topic_words = self.bertopic_model.get_topic(row['Topic'])
                keywords = [{'word': word, 'weight': weight} for word, weight in topic_words[:5]]
                
                topics[f"주제_{row['Topic']+1}"] = {
                    'keywords': keywords,
                    'count': row['Count'],
                    'description': f"BERTopic 주제 {row['Topic']+1}"
                }
        else:
            # 간단한 토픽 모델링 결과
            if hasattr(self, '_simple_topics'):
                for idx, topic_words in enumerate(self._simple_topics):
                    keywords = [{'word': word, 'weight': count} for word, count in topic_words]
                    topics[f"주제_{idx+1}"] = {
                        'keywords': keywords,
                        'description': f"Simple Topic {idx+1}"
                    }
        
        self.results['bertopic_topics'] = topics
        return topics
    
    def analyze_documents(self, texts, categories=None):
        """문서별 토픽 분석"""
        print("문서별 토픽 분석...")
        
        doc_topics = []
        
        if self.lda_model and self.vectorizer:
            # Scikit-learn LDA의 경우
            try:
                text_strings = [str(text) for text in texts if text]
                tfidf_matrix = self.vectorizer.transform(text_strings)
                doc_topic_probs = self.lda_model.transform(tfidf_matrix)
                
                for i, probs in enumerate(doc_topic_probs):
                    dominant_topic = np.argmax(probs)
                    max_prob = probs[dominant_topic]
                    
                    doc_topics.append({
                        'document_id': i,
                        'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                        'dominant_topic': int(dominant_topic + 1),
                        'probability': float(max_prob),
                        'category': categories[i] if categories else 'unknown'
                    })
            except Exception as e:
                print(f"문서 토픽 분석 실패: {e}")
        
        self.results['document_topics'] = doc_topics
        return doc_topics
    
    def compare_methods(self):
        """LDA와 BERTopic 비교 분석"""
        lda_topics = self.results.get('lda_topics', {})
        bertopic_topics = self.results.get('bertopic_topics', {})
        
        comparison = {
            'method_comparison': {
                'LDA': {
                    'topics_count': len(lda_topics),
                    'method': 'Statistical topic modeling',
                    'advantages': ['빠른 학습', '해석 용이', '안정적 결과'],
                    'disadvantages': ['단어 순서 무시', '의미 표현 한계']
                },
                'BERTopic': {
                    'topics_count': len(bertopic_topics),
                    'method': 'Transformer-based topic modeling',
                    'advantages': ['의미적 유사성', '동적 토픽 수', '고품질 임베딩'],
                    'disadvantages': ['높은 계산 비용', '복잡한 구현']
                }
            },
            'topic_overlap': self.calculate_topic_overlap(lda_topics, bertopic_topics)
        }
        
        self.results['topic_comparison'] = comparison
        return comparison
    
    def calculate_topic_overlap(self, lda_topics, bertopic_topics):
        """토픽 간 중복도 계산"""
        overlaps = []
        
        for lda_key, lda_topic in lda_topics.items():
            lda_words = set([kw['word'] for kw in lda_topic['keywords']])
            
            for bert_key, bert_topic in bertopic_topics.items():
                bert_words = set([kw['word'] for kw in bert_topic['keywords']])
                
                intersection = lda_words.intersection(bert_words)
                union = lda_words.union(bert_words)
                
                if union:
                    jaccard_similarity = len(intersection) / len(union)
                    if jaccard_similarity > 0.2:  # 20% 이상 유사도
                        overlaps.append({
                            'lda_topic': lda_key,
                            'bertopic_topic': bert_key,
                            'similarity': jaccard_similarity,
                            'common_words': list(intersection)
                        })
        
        return overlaps
    
    def visualize_topics(self):
        """토픽 시각화"""
        output_dir = 'C:/Dev/book-analysis/practice/chapter04/output'
        os.makedirs(output_dir, exist_ok=True)
        
        # LDA 토픽 워드클라우드 대체: 키워드 빈도 차트
        lda_topics = self.results.get('lda_topics', {})
        
        if lda_topics:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle('LDA 주제별 주요 키워드', fontsize=16)
            
            for idx, (topic_name, topic_data) in enumerate(lda_topics.items()):
                if idx >= 10:
                    break
                    
                row = idx // 5
                col = idx % 5
                
                keywords = [kw['word'] for kw in topic_data['keywords']]
                weights = [kw['weight'] for kw in topic_data['keywords']]
                
                axes[row, col].barh(keywords, weights)
                axes[row, col].set_title(f'{topic_name}', fontsize=12)
                axes[row, col].tick_params(labelsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/lda_topics.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 토픽 분포 히트맵
        doc_topics = self.results.get('document_topics', [])
        if doc_topics:
            topic_counts = defaultdict(int)
            for doc in doc_topics:
                topic_counts[doc['dominant_topic']] += 1
            
            plt.figure(figsize=(12, 6))
            topics = list(topic_counts.keys())
            counts = list(topic_counts.values())
            
            plt.bar(topics, counts)
            plt.title('문서별 주제 분포')
            plt.xlabel('주제 번호')
            plt.ylabel('문서 수')
            plt.xticks(topics)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/topic_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"시각화 결과 저장: {output_dir}")
    
    def save_results(self, output_path):
        """결과 저장"""
        # 분석 메타데이터 추가
        self.results['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'num_topics': self.num_topics,
            'lda_available': GENSIM_AVAILABLE,
            'bertopic_available': BERTOPIC_AVAILABLE,
            'total_documents': len(self.results.get('document_topics', []))
        }
        
        # JSON 직렬화를 위해 numpy 타입 변환
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 결과를 JSON으로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        print(f"분석 결과 저장: {output_path}")
        return self.results

def main():
    """메인 실행 함수"""
    
    print("=" * 50)
    print("LDA & BERTopic 토픽 모델링 분석")
    print("=" * 50)
    
    # 전처리된 데이터 로드
    data_path = 'C:/Dev/book-analysis/practice/chapter04/data/preprocessed_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    print(f"\n데이터 로드 완료: {len(df)}개 문서")
    
    # 분석기 초기화
    analyzer = TopicAnalyzer(num_topics=10)
    
    # 전처리된 텍스트 추출
    texts = df['processed_text'].tolist()
    categories = df['category'].tolist()
    
    # LDA 모델 학습
    print("\n1. LDA 토픽 모델링...")
    lda_model = analyzer.train_lda_model(texts)
    
    if lda_model:
        lda_topics = analyzer.extract_lda_topics()
        print(f"LDA 주제 추출 완료: {len(lda_topics)}개 주제")
        
        # 주제 미리보기
        for topic_name, topic_data in list(lda_topics.items())[:3]:
            keywords = [kw['word'] for kw in topic_data['keywords']]
            print(f"  {topic_name}: {', '.join(keywords)}")
    
    # BERTopic 모델 학습
    print("\n2. BERTopic 토픽 모델링...")
    bertopic_result = analyzer.train_bertopic_model(texts)
    
    if bertopic_result:
        # BERTopic 결과 처리
        if isinstance(bertopic_result, tuple):
            topics, probabilities = bertopic_result
            if hasattr(analyzer, 'bertopic_model'):
                bertopic_topics = analyzer.extract_bertopic_topics()
            else:
                # 간단한 토픽 모델링 결과 저장
                analyzer._simple_topics = topics
                bertopic_topics = analyzer.extract_bertopic_topics()
        else:
            bertopic_topics = analyzer.extract_bertopic_topics()
        
        print(f"BERTopic 주제 추출 완료: {len(bertopic_topics)}개 주제")
        
        # 주제 미리보기
        for topic_name, topic_data in list(bertopic_topics.items())[:3]:
            keywords = [kw['word'] for kw in topic_data['keywords']]
            print(f"  {topic_name}: {', '.join(keywords)}")
    
    # 문서별 토픽 분석
    print("\n3. 문서별 토픽 분석...")
    doc_topics = analyzer.analyze_documents(texts, categories)
    print(f"문서별 토픽 분석 완료: {len(doc_topics)}개 문서")
    
    # 방법론 비교
    print("\n4. 방법론 비교 분석...")
    comparison = analyzer.compare_methods()
    
    # 시각화
    print("\n5. 결과 시각화...")
    try:
        analyzer.visualize_topics()
    except Exception as e:
        print(f"시각화 실패: {e}")
    
    # 결과 저장
    output_path = 'C:/Dev/book-analysis/practice/chapter04/output/topic_analysis.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = analyzer.save_results(output_path)
    
    # 결과 요약 출력
    print("\n" + "=" * 50)
    print("토픽 모델링 분석 결과 요약")
    print("=" * 50)
    
    print(f"분석 문서 수: {len(df)}")
    print(f"추출된 주제 수: {analyzer.num_topics}")
    print(f"LDA 주제: {len(results.get('lda_topics', {}))}")
    print(f"BERTopic 주제: {len(results.get('bertopic_topics', {}))}")
    
    # 주요 주제 키워드 출력
    print("\n주요 주제 키워드:")
    lda_topics = results.get('lda_topics', {})
    for topic_name, topic_data in list(lda_topics.items())[:5]:
        keywords = [kw['word'] for kw in topic_data['keywords'][:3]]
        print(f"  {topic_name}: {', '.join(keywords)}")
    
    # 카테고리별 주제 분포
    if doc_topics:
        category_topics = defaultdict(list)
        for doc in doc_topics:
            category_topics[doc['category']].append(doc['dominant_topic'])
        
        print("\n카테고리별 주요 주제:")
        for category, topics_list in category_topics.items():
            most_common_topic = Counter(topics_list).most_common(1)[0]
            print(f"  {category}: 주제 {most_common_topic[0]} ({most_common_topic[1]}건)")
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()