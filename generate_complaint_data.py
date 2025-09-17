"""
제4장: 정책 텍스트 분석을 위한 민원 데이터 생성
국민신문고 및 에스토니아 민원 데이터의 특성을 반영한 현실적인 샘플 데이터 생성
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# 재현성을 위한 시드 고정
np.random.seed(42)
random.seed(42)

def generate_korean_complaints():
    """한국 국민신문고 스타일의 민원 데이터 생성"""
    
    # 민원 카테고리 및 템플릿
    categories = {
        '복지': {
            'keywords': ['기초연금', '복지수당', '지원금', '생활보조', '의료비', '보육료'],
            'templates': [
                "기초연금 수급 자격 기준이 너무 엄격합니다. {detail}",
                "복지 지원금 신청 절차가 복잡하여 어려움을 겪고 있습니다. {detail}",
                "의료비 지원 대상자 선정 기준을 완화해 주세요. {detail}",
                "보육료 지원이 실제 필요한 가정에 제대로 전달되지 않습니다. {detail}",
                "노인 복지 서비스 확대가 필요합니다. {detail}"
            ],
            'details': [
                "소득 기준을 현실화해야 합니다.",
                "온라인 신청 시스템 개선이 시급합니다.",
                "서류 준비 과정이 너무 복잡합니다.",
                "담당 공무원의 안내가 부족합니다.",
                "지역별 격차가 심각합니다."
            ]
        },
        '교통': {
            'keywords': ['버스', '지하철', '도로', '신호등', '주차', '교통체증'],
            'templates': [
                "출퇴근 시간 {location} 버스 배차 간격이 너무 깁니다. {detail}",
                "{location} 지역 신호등 체계 개선이 필요합니다. {detail}",
                "주차 공간 부족 문제를 해결해 주세요. {detail}",
                "{location} 도로 포장 상태가 매우 불량합니다. {detail}",
                "대중교통 환승 시스템 개선이 필요합니다. {detail}"
            ],
            'details': [
                "시민들의 불편이 가중되고 있습니다.",
                "사고 위험이 높아지고 있습니다.",
                "경제적 손실이 발생하고 있습니다.",
                "긴급한 대책 마련이 필요합니다.",
                "장기적인 계획 수립을 요청합니다."
            ]
        },
        '환경': {
            'keywords': ['미세먼지', '소음', '쓰레기', '재활용', '공원', '녹지'],
            'templates': [
                "{location} 지역 미세먼지 대책을 강화해 주세요. {detail}",
                "생활 소음 규제를 강화해야 합니다. {detail}",
                "재활용 분리수거 시스템 개선이 필요합니다. {detail}",
                "{location} 공원 관리가 제대로 되지 않고 있습니다. {detail}",
                "도시 녹지 공간을 확대해 주세요. {detail}"
            ],
            'details': [
                "주민 건강이 위협받고 있습니다.",
                "생활 환경이 악화되고 있습니다.",
                "체계적인 관리가 필요합니다.",
                "예산 확대가 시급합니다.",
                "시민 참여 방안을 마련해 주세요."
            ]
        },
        '행정': {
            'keywords': ['민원처리', '공무원', '서류발급', '온라인시스템', '행정절차'],
            'templates': [
                "민원 처리 속도가 너무 느립니다. {detail}",
                "온라인 행정 시스템이 불편합니다. {detail}",
                "서류 발급 절차를 간소화해 주세요. {detail}",
                "공무원 응대 서비스 개선이 필요합니다. {detail}",
                "행정 정보 공개를 확대해 주세요. {detail}"
            ],
            'details': [
                "디지털 전환이 시급합니다.",
                "업무 효율성을 높여야 합니다.",
                "시민 중심 서비스가 필요합니다.",
                "투명성을 강화해야 합니다.",
                "접근성을 개선해 주세요."
            ]
        },
        '안전': {
            'keywords': ['CCTV', '가로등', '범죄예방', '재난대응', '안전시설'],
            'templates': [
                "{location} 지역에 CCTV 설치를 요청합니다. {detail}",
                "가로등이 고장난 지 오래되었습니다. {detail}",
                "어린이 보호구역 안전시설을 강화해 주세요. {detail}",
                "재난 대응 시스템 개선이 필요합니다. {detail}",
                "공공시설 안전 점검을 철저히 해주세요. {detail}"
            ],
            'details': [
                "주민 불안감이 증가하고 있습니다.",
                "사고 예방이 시급합니다.",
                "신속한 조치를 요청합니다.",
                "정기적인 점검이 필요합니다.",
                "예방 대책을 마련해 주세요."
            ]
        }
    }
    
    locations = ['서울 강남구', '서울 강북구', '부산 해운대구', '대구 수성구', '인천 연수구', 
                 '광주 서구', '대전 유성구', '울산 남구', '세종시', '경기 수원시', 
                 '경기 성남시', '경기 용인시', '강원 춘천시', '충북 청주시', '충남 천안시',
                 '전북 전주시', '전남 여수시', '경북 포항시', '경남 창원시', '제주시']
    
    complaints = []
    
    for i in range(700):  # 한국 민원 700건
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]
        
        template = random.choice(cat_data['templates'])
        detail = random.choice(cat_data['details'])
        location = random.choice(locations)
        
        # 민원 텍스트 생성
        complaint_text = template.format(location=location, detail=detail)
        
        # 감성 레이블 (대부분 부정적이지만 일부 중립/긍정)
        sentiment_prob = random.random()
        if sentiment_prob < 0.7:
            sentiment = 'negative'
        elif sentiment_prob < 0.9:
            sentiment = 'neutral'
        else:
            sentiment = 'positive'
        
        # 처리 상태
        status_prob = random.random()
        if status_prob < 0.3:
            status = '접수'
        elif status_prob < 0.7:
            status = '처리중'
        else:
            status = '완료'
        
        # 날짜 생성 (최근 1년)
        days_ago = random.randint(0, 365)
        complaint_date = datetime.now() - timedelta(days=days_ago)
        
        complaints.append({
            'complaint_id': f'KR_{i+1:05d}',
            'text': complaint_text,
            'category': category,
            'sentiment': sentiment,
            'status': status,
            'date': complaint_date.strftime('%Y-%m-%d'),
            'location': location,
            'language': 'ko',
            'source': '국민신문고'
        })
    
    return complaints

def generate_estonian_complaints():
    """에스토니아 스타일의 민원 데이터 생성 (영어)"""
    
    categories = {
        'Digital Services': {
            'keywords': ['e-government', 'digital ID', 'online service', 'portal', 'authentication'],
            'templates': [
                "The e-government portal is experiencing technical issues. {detail}",
                "Digital ID authentication process needs improvement. {detail}",
                "Online service for {service} is not user-friendly. {detail}",
                "Mobile app functionality should be enhanced. {detail}",
                "Data synchronization between services is problematic. {detail}"
            ],
            'details': [
                "The system response time is too slow.",
                "User interface needs modernization.",
                "Error messages are not informative.",
                "Mobile compatibility issues persist.",
                "Integration with other services required."
            ],
            'services': ['tax filing', 'permit applications', 'voting', 'healthcare booking', 'education enrollment']
        },
        'Public Transport': {
            'keywords': ['bus', 'tram', 'schedule', 'route', 'ticket'],
            'templates': [
                "Public transport schedule in {area} needs adjustment. {detail}",
                "Electronic ticketing system has issues. {detail}",
                "Bus route optimization required for {area}. {detail}",
                "Real-time tracking system not accurate. {detail}",
                "Accessibility features need improvement. {detail}"
            ],
            'details': [
                "Commuters face daily delays.",
                "System updates are urgently needed.",
                "Better coordination required.",
                "Information display is inadequate.",
                "Service frequency is insufficient."
            ]
        },
        'Healthcare': {
            'keywords': ['appointment', 'e-health', 'prescription', 'medical records', 'insurance'],
            'templates': [
                "E-health system appointment booking is complicated. {detail}",
                "Digital prescription system needs enhancement. {detail}",
                "Medical records access should be simplified. {detail}",
                "Insurance claim process is too complex. {detail}",
                "Telemedicine services require expansion. {detail}"
            ],
            'details': [
                "Elderly citizens struggle with the system.",
                "Technical support is inadequate.",
                "Processing time is excessive.",
                "System integration issues exist.",
                "User guidance is insufficient."
            ]
        },
        'Education': {
            'keywords': ['e-school', 'digital learning', 'enrollment', 'curriculum', 'resources'],
            'templates': [
                "E-school platform needs technical improvements. {detail}",
                "Digital learning resources are outdated. {detail}",
                "School enrollment system is confusing. {detail}",
                "Parent portal functionality is limited. {detail}",
                "Student data management needs attention. {detail}"
            ],
            'details': [
                "Teachers need better training.",
                "System crashes during peak times.",
                "Content quality varies significantly.",
                "Communication features are lacking.",
                "Mobile access is problematic."
            ]
        },
        'Environment': {
            'keywords': ['recycling', 'waste', 'green spaces', 'pollution', 'sustainability'],
            'templates': [
                "Waste management system in {area} needs improvement. {detail}",
                "Recycling facilities are insufficient. {detail}",
                "Green spaces maintenance is inadequate. {detail}",
                "Air quality monitoring should be enhanced. {detail}",
                "Sustainable energy initiatives needed. {detail}"
            ],
            'details': [
                "Residents demand better services.",
                "Environmental impact is concerning.",
                "Investment in infrastructure required.",
                "Public awareness campaigns needed.",
                "Long-term planning is essential."
            ]
        }
    }
    
    areas = ['Tallinn', 'Tartu', 'Narva', 'Pärnu', 'Kohtla-Järve', 'Viljandi', 'Rakvere', 
             'Maardu', 'Kuressaare', 'Sillamäe', 'Võru', 'Valga', 'Jõhvi', 'Haapsalu', 'Keila']
    
    complaints = []
    
    for i in range(300):  # 에스토니아 민원 300건
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]
        
        template = random.choice(cat_data['templates'])
        detail = random.choice(cat_data['details'])
        area = random.choice(areas)
        
        # 템플릿에 서비스가 필요한 경우
        if '{service}' in template and 'services' in cat_data:
            service = random.choice(cat_data['services'])
            complaint_text = template.format(area=area, service=service, detail=detail)
        else:
            complaint_text = template.format(area=area, detail=detail)
        
        # 감성 레이블
        sentiment_prob = random.random()
        if sentiment_prob < 0.6:
            sentiment = 'negative'
        elif sentiment_prob < 0.85:
            sentiment = 'neutral'
        else:
            sentiment = 'positive'
        
        # 처리 상태
        status_prob = random.random()
        if status_prob < 0.25:
            status = 'submitted'
        elif status_prob < 0.65:
            status = 'in_progress'
        else:
            status = 'resolved'
        
        # 날짜 생성
        days_ago = random.randint(0, 365)
        complaint_date = datetime.now() - timedelta(days=days_ago)
        
        complaints.append({
            'complaint_id': f'EE_{i+1:05d}',
            'text': complaint_text,
            'category': category,
            'sentiment': sentiment,
            'status': status,
            'date': complaint_date.strftime('%Y-%m-%d'),
            'location': area,
            'language': 'en',
            'source': 'Estonian e-Gov Portal'
        })
    
    return complaints

def main():
    """메인 실행 함수"""
    print("민원 데이터 생성 시작...")
    
    # 한국 민원 데이터 생성
    korean_complaints = generate_korean_complaints()
    print(f"한국 민원 데이터 {len(korean_complaints)}건 생성 완료")
    
    # 에스토니아 민원 데이터 생성
    estonian_complaints = generate_estonian_complaints()
    print(f"에스토니아 민원 데이터 {len(estonian_complaints)}건 생성 완료")
    
    # 전체 데이터 합치기
    all_complaints = korean_complaints + estonian_complaints
    random.shuffle(all_complaints)  # 데이터 섞기
    
    # DataFrame 생성
    df = pd.DataFrame(all_complaints)
    
    # 데이터 저장
    output_path = 'C:/Dev/book-analysis/practice/chapter04/data/complaints_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n전체 민원 데이터 {len(df)}건을 {output_path}에 저장했습니다.")
    
    # 데이터 요약 정보
    print("\n=== 데이터 요약 ===")
    print(f"총 민원 수: {len(df)}")
    print(f"\n언어별 분포:")
    print(df['language'].value_counts())
    print(f"\n카테고리별 분포:")
    print(df['category'].value_counts())
    print(f"\n감성 분포:")
    print(df['sentiment'].value_counts())
    print(f"\n처리 상태 분포:")
    print(df['status'].value_counts())
    
    # 샘플 데이터 출력
    print("\n=== 샘플 데이터 (처음 5개) ===")
    print(df.head())
    
    # 데이터 코드북 생성
    codebook = {
        'dataset_info': {
            'name': '정책 민원 텍스트 분석 데이터셋',
            'description': '국민신문고 및 에스토니아 e-Gov 포털 스타일의 민원 데이터',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'total_records': len(df),
            'languages': ['Korean (ko)', 'English (en)'],
            'sources': ['국민신문고', 'Estonian e-Gov Portal']
        },
        'variables': {
            'complaint_id': {
                'type': 'string',
                'description': '민원 고유 ID',
                'format': 'KR_##### 또는 EE_#####'
            },
            'text': {
                'type': 'string',
                'description': '민원 내용 텍스트',
                'language': '한국어 또는 영어'
            },
            'category': {
                'type': 'categorical',
                'description': '민원 카테고리',
                'values': list(df['category'].unique())
            },
            'sentiment': {
                'type': 'categorical',
                'description': '감성 레이블',
                'values': ['positive', 'neutral', 'negative']
            },
            'status': {
                'type': 'categorical',
                'description': '처리 상태',
                'values': list(df['status'].unique())
            },
            'date': {
                'type': 'date',
                'description': '민원 접수일',
                'format': 'YYYY-MM-DD'
            },
            'location': {
                'type': 'string',
                'description': '민원 발생 지역'
            },
            'language': {
                'type': 'categorical',
                'description': '언어 코드',
                'values': ['ko', 'en']
            },
            'source': {
                'type': 'categorical',
                'description': '데이터 출처',
                'values': ['국민신문고', 'Estonian e-Gov Portal']
            }
        }
    }
    
    # 코드북 저장
    codebook_path = 'C:/Dev/book-analysis/practice/chapter04/data/codebook.json'
    with open(codebook_path, 'w', encoding='utf-8') as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)
    print(f"\n코드북을 {codebook_path}에 저장했습니다.")
    
    return df

if __name__ == "__main__":
    df = main()