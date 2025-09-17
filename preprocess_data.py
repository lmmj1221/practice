"""
제4장: 민원 데이터 전처리 파이프라인
한국어와 영어 민원 텍스트의 전처리 및 정제
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한국어 처리를 위한 라이브러리
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    print("Warning: KoNLPy not installed. Korean text processing will be limited.")

# 영어 처리를 위한 라이브러리
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# NLTK 데이터 다운로드 (필요시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 재현성을 위한 시드 고정
np.random.seed(42)

class ComplaintPreprocessor:
    """민원 텍스트 전처리 클래스"""
    
    def __init__(self):
        """전처리기 초기화"""
        # 한국어 처리기
        if KONLPY_AVAILABLE:
            self.okt = Okt()
        
        # 영어 처리기
        self.stemmer = PorterStemmer()
        self.english_stopwords = set(stopwords.words('english'))
        
        # 한국어 불용어
        self.korean_stopwords = {
            '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', 
            '를', '으로', '자', '에', '와', '한', '하다', '을', '를', '이다',
            '있다', '되다', '수', '보다', '없다', '않다', '그', '때문', '그리고',
            '그러나', '그런데', '하지만', '따라서', '그래서', '그러므로'
        }
        
        # 전처리 통계
        self.stats = {
            'total_processed': 0,
            'korean_processed': 0,
            'english_processed': 0,
            'avg_length_before': 0,
            'avg_length_after': 0,
            'processing_time': None
        }
    
    def clean_text(self, text):
        """텍스트 기본 정제"""
        if pd.isna(text):
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL 제거
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 이메일 제거
        text = re.sub(r'\S+@\S+', '', text)
        
        # 특수문자를 공백으로 치환 (한글, 영문, 숫자 제외)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def preprocess_korean(self, text):
        """한국어 텍스트 전처리"""
        # 기본 정제
        text = self.clean_text(text)
        
        if not text or not KONLPY_AVAILABLE:
            return text
        
        try:
            # 형태소 분석
            morphs = self.okt.morphs(text)
            
            # 불용어 제거
            filtered_morphs = [word for word in morphs if word not in self.korean_stopwords]
            
            # 토큰이 너무 짧은 경우 제거 (1글자)
            filtered_morphs = [word for word in filtered_morphs if len(word) > 1]
            
            # 다시 합치기
            processed_text = ' '.join(filtered_morphs)
            
            return processed_text
        
        except Exception as e:
            print(f"Korean preprocessing error: {e}")
            return text
    
    def preprocess_english(self, text):
        """영어 텍스트 전처리"""
        # 기본 정제
        text = self.clean_text(text)
        
        if not text:
            return text
        
        try:
            # 토큰화
            tokens = word_tokenize(text)
            
            # 불용어 제거
            filtered_tokens = [word for word in tokens if word not in self.english_stopwords]
            
            # 스테밍
            stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
            
            # 토큰이 너무 짧은 경우 제거 (2글자 미만)
            filtered_tokens = [word for word in stemmed_tokens if len(word) >= 2]
            
            # 다시 합치기
            processed_text = ' '.join(filtered_tokens)
            
            return processed_text
        
        except Exception as e:
            print(f"English preprocessing error: {e}")
            return text
    
    def preprocess_dataframe(self, df):
        """데이터프레임 전체 전처리"""
        start_time = datetime.now()
        
        print("데이터 전처리 시작...")
        print(f"전체 레코드 수: {len(df)}")
        
        # 원본 텍스트 백업
        df['original_text'] = df['text']
        
        # 전처리 전 평균 길이
        self.stats['avg_length_before'] = df['text'].str.len().mean()
        
        # 언어별 전처리 적용
        preprocessed_texts = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"처리 중... {idx}/{len(df)}")
            
            if row['language'] == 'ko':
                processed_text = self.preprocess_korean(row['text'])
                self.stats['korean_processed'] += 1
            else:
                processed_text = self.preprocess_english(row['text'])
                self.stats['english_processed'] += 1
            
            preprocessed_texts.append(processed_text)
            self.stats['total_processed'] += 1
        
        df['processed_text'] = preprocessed_texts
        
        # 전처리 후 평균 길이
        self.stats['avg_length_after'] = df['processed_text'].str.len().mean()
        
        # 토큰 수 계산
        df['token_count'] = df['processed_text'].str.split().str.len()
        
        # 빈 텍스트 확인
        empty_count = df['processed_text'].str.strip().eq('').sum()
        if empty_count > 0:
            print(f"Warning: {empty_count} empty texts after preprocessing")
        
        # 처리 시간
        end_time = datetime.now()
        self.stats['processing_time'] = str(end_time - start_time)
        
        print("전처리 완료!")
        
        return df
    
    def generate_statistics(self, df):
        """전처리 통계 생성"""
        stats = {
            '전처리_통계': self.stats,
            '텍스트_길이_분포': {
                '원본_평균': round(df['original_text'].str.len().mean(), 2),
                '원본_중앙값': df['original_text'].str.len().median(),
                '처리후_평균': round(df['processed_text'].str.len().mean(), 2),
                '처리후_중앙값': df['processed_text'].str.len().median(),
                '압축률': round((1 - df['processed_text'].str.len().mean() / 
                               df['original_text'].str.len().mean()) * 100, 2)
            },
            '토큰_통계': {
                '평균_토큰수': round(df['token_count'].mean(), 2),
                '최대_토큰수': df['token_count'].max(),
                '최소_토큰수': df['token_count'].min(),
                '중앙값_토큰수': df['token_count'].median()
            },
            '언어별_분포': df['language'].value_counts().to_dict(),
            '카테고리별_평균_토큰수': df.groupby('category')['token_count'].mean().round(2).to_dict()
        }
        
        return stats

def create_codebook(df, stats):
    """데이터 코드북 생성"""
    codebook = {
        '데이터셋_정보': {
            '이름': '전처리된 민원 텍스트 데이터셋',
            '설명': 'BERT/KoBERT 분석을 위한 전처리된 민원 데이터',
            '전처리_일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '총_레코드수': len(df),
            '전처리_방법': {
                '한국어': 'KoNLPy Okt 형태소 분석, 불용어 제거',
                '영어': 'NLTK 토큰화, 불용어 제거, Porter Stemming'
            }
        },
        '변수_설명': {
            'complaint_id': '민원 고유 ID',
            'original_text': '원본 민원 텍스트',
            'processed_text': '전처리된 민원 텍스트',
            'token_count': '토큰 개수',
            'category': '민원 카테고리',
            'sentiment': '감성 레이블 (positive/neutral/negative)',
            'status': '처리 상태',
            'date': '민원 접수일',
            'location': '민원 발생 지역',
            'language': '언어 코드 (ko/en)',
            'source': '데이터 출처'
        },
        '전처리_통계': stats,
        '데이터_품질': {
            '결측값': df.isnull().sum().to_dict(),
            '중복_레코드': len(df) - len(df.drop_duplicates()),
            '빈_텍스트': df['processed_text'].str.strip().eq('').sum()
        }
    }
    
    return codebook

def main():
    """메인 실행 함수"""
    # 데이터 로드
    input_path = 'C:/Dev/book-analysis/practice/chapter04/data/complaints_data.csv'
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    
    print("=" * 50)
    print("민원 데이터 전처리 파이프라인")
    print("=" * 50)
    
    # 전처리기 생성
    preprocessor = ComplaintPreprocessor()
    
    # 데이터 전처리
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # 통계 생성
    stats = preprocessor.generate_statistics(df_processed)
    
    # 전처리된 데이터 저장
    output_path = 'C:/Dev/book-analysis/practice/chapter04/data/preprocessed_data.csv'
    df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n전처리된 데이터를 {output_path}에 저장했습니다.")
    
    # 코드북 생성 및 저장
    codebook = create_codebook(df_processed, stats)
    
    # JSON 형식으로 저장
    codebook_json_path = 'C:/Dev/book-analysis/practice/chapter04/data/preprocessed_codebook.json'
    with open(codebook_json_path, 'w', encoding='utf-8') as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2, default=str)
    print(f"코드북(JSON)을 {codebook_json_path}에 저장했습니다.")
    
    # Markdown 형식으로도 저장
    codebook_md_path = 'C:/Dev/book-analysis/practice/chapter04/data/codebook.md'
    with open(codebook_md_path, 'w', encoding='utf-8') as f:
        f.write("# 전처리된 민원 데이터 코드북\n\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 데이터셋 정보\n")
        f.write(f"- 총 레코드 수: {len(df_processed)}\n")
        f.write(f"- 한국어 민원: {stats['전처리_통계']['korean_processed']}건\n")
        f.write(f"- 영어 민원: {stats['전처리_통계']['english_processed']}건\n")
        f.write(f"- 처리 시간: {stats['전처리_통계']['processing_time']}\n\n")
        
        f.write("## 텍스트 통계\n")
        f.write(f"- 원본 텍스트 평균 길이: {stats['텍스트_길이_분포']['원본_평균']}자\n")
        f.write(f"- 전처리 후 평균 길이: {stats['텍스트_길이_분포']['처리후_평균']}자\n")
        f.write(f"- 텍스트 압축률: {stats['텍스트_길이_분포']['압축률']}%\n")
        f.write(f"- 평균 토큰 수: {stats['토큰_통계']['평균_토큰수']}\n\n")
        
        f.write("## 변수 설명\n")
        for var, desc in codebook['변수_설명'].items():
            f.write(f"- **{var}**: {desc}\n")
    
    print(f"코드북(Markdown)을 {codebook_md_path}에 저장했습니다.")
    
    # 통계 출력
    print("\n" + "=" * 50)
    print("전처리 통계 요약")
    print("=" * 50)
    print(f"총 처리 레코드: {stats['전처리_통계']['total_processed']}")
    print(f"텍스트 압축률: {stats['텍스트_길이_분포']['압축률']}%")
    print(f"평균 토큰 수: {stats['토큰_통계']['평균_토큰수']}")
    print(f"처리 시간: {stats['전처리_통계']['processing_time']}")
    
    # 샘플 출력
    print("\n" + "=" * 50)
    print("전처리 샘플 (처음 3개)")
    print("=" * 50)
    for idx in range(min(3, len(df_processed))):
        row = df_processed.iloc[idx]
        print(f"\n[{idx+1}] ID: {row['complaint_id']}")
        print(f"원본: {row['original_text'][:100]}...")
        print(f"처리: {row['processed_text'][:100]}...")
        print(f"토큰 수: {row['token_count']}")
    
    return df_processed

if __name__ == "__main__":
    df = main()