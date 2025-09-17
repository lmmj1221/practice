# 예제 1.5: 참고문헌 자동 정리 스크립트
import json
import re
from datetime import datetime

class ReferenceFormatter:
    """참고문헌 자동 정리 시스템"""
    
    def __init__(self):
        self.references = []
        
    def add_reference(self, ref_type, **kwargs):
        """참고문헌 추가"""
        ref = {
            'type': ref_type,
            'added_date': datetime.now().strftime('%Y-%m-%d'),
            **kwargs
        }
        self.references.append(ref)
        
    def to_apa(self, ref):
        """APA 형식으로 변환"""
        if ref['type'] == 'journal':
            # Author, A. A. (Year). Title. Journal, Volume(Issue), pages.
            authors = ' & '.join(ref.get('authors', []))
            return f"{authors} ({ref.get('year', 'n.d.')}). {ref.get('title', 'Untitled')}. " \
                   f"{ref.get('journal', 'Unknown Journal')}, {ref.get('volume', '')}" \
                   f"({ref.get('issue', '')}), {ref.get('pages', '')}."
                   
        elif ref['type'] == 'book':
            # Author, A. A. (Year). Title. Publisher.
            authors = ' & '.join(ref.get('authors', []))
            return f"{authors} ({ref.get('year', 'n.d.')}). {ref.get('title', 'Untitled')}. " \
                   f"{ref.get('publisher', 'Unknown Publisher')}."
                   
        elif ref['type'] == 'website':
            # Author. (Year). Title. Retrieved from URL
            author = ref.get('author', 'Unknown Author')
            return f"{author}. ({ref.get('year', 'n.d.')}). {ref.get('title', 'Untitled')}. " \
                   f"Retrieved from {ref.get('url', '')}"
                   
        elif ref['type'] == 'government':
            # Organization. (Year). Title (Report No. xxx). Publisher.
            org = ref.get('organization', 'Unknown Organization')
            report_no = f"(Report No. {ref.get('report_number')}). " if ref.get('report_number') else ''
            return f"{org}. ({ref.get('year', 'n.d.')}). {ref.get('title', 'Untitled')} " \
                   f"{report_no}{ref.get('publisher', '')}."
        else:
            return str(ref)
    
    def to_bibtex(self, ref):
        """BibTeX 형식으로 변환"""
        # Generate citation key
        first_author = ref.get('authors', ['Unknown'])[0].split(',')[0].replace(' ', '')
        year = ref.get('year', 'XXXX')
        key = f"{first_author}{year}"
        
        if ref['type'] == 'journal':
            return f"""@article{{{key},
  author = {{{' and '.join(ref.get('authors', []))}}},
  title = {{{ref.get('title', 'Untitled')}}},
  journal = {{{ref.get('journal', 'Unknown Journal')}}},
  year = {{{ref.get('year', 'n.d.')}}},
  volume = {{{ref.get('volume', '')}}},
  number = {{{ref.get('issue', '')}}},
  pages = {{{ref.get('pages', '')}}}
}}"""
        elif ref['type'] == 'book':
            return f"""@book{{{key},
  author = {{{' and '.join(ref.get('authors', []))}}},
  title = {{{ref.get('title', 'Untitled')}}},
  publisher = {{{ref.get('publisher', 'Unknown Publisher')}}},
  year = {{{ref.get('year', 'n.d.')}}}
}}"""
        elif ref['type'] == 'website':
            return f"""@misc{{{key},
  author = {{{ref.get('author', 'Unknown Author')}}},
  title = {{{ref.get('title', 'Untitled')}}},
  year = {{{ref.get('year', 'n.d.')}}},
  url = {{{ref.get('url', '')}}},
  note = {{Accessed: {datetime.now().strftime('%Y-%m-%d')}}}
}}"""
        elif ref['type'] == 'government':
            return f"""@techreport{{{key},
  author = {{{ref.get('organization', 'Unknown Organization')}}},
  title = {{{ref.get('title', 'Untitled')}}},
  year = {{{ref.get('year', 'n.d.')}}},
  institution = {{{ref.get('publisher', '')}}},
  number = {{{ref.get('report_number', '')}}}
}}"""
        else:
            return f"% Unsupported type: {ref['type']}"
    
    def format_all(self, format_type='apa'):
        """모든 참고문헌 포맷팅"""
        formatted = []
        for ref in sorted(self.references, key=lambda x: (x.get('year', '9999'), x.get('title', ''))):
            if format_type == 'apa':
                formatted.append(self.to_apa(ref))
            elif format_type == 'bibtex':
                formatted.append(self.to_bibtex(ref))
        return formatted
    
    def save_to_file(self, filename, format_type='apa'):
        """파일로 저장"""
        formatted = self.format_all(format_type)
        with open(filename, 'w', encoding='utf-8') as f:
            if format_type == 'apa':
                f.write("# References (APA Format)\n\n")
                for i, ref in enumerate(formatted, 1):
                    f.write(f"{i}. {ref}\n\n")
            elif format_type == 'bibtex':
                f.write("% BibTeX References\n\n")
                for ref in formatted:
                    f.write(ref + "\n\n")
        print(f"References saved to {filename}")
    
    def analyze_references(self):
        """참고문헌 통계 분석"""
        stats = {
            'total': len(self.references),
            'by_type': {},
            'by_year': {},
            'recent_5_years': 0
        }
        
        current_year = datetime.now().year
        
        for ref in self.references:
            # Type statistics
            ref_type = ref['type']
            stats['by_type'][ref_type] = stats['by_type'].get(ref_type, 0) + 1
            
            # Year statistics
            year = ref.get('year', 'Unknown')
            if year != 'Unknown':
                stats['by_year'][year] = stats['by_year'].get(year, 0) + 1
                
                # Recent 5 years
                try:
                    if current_year - int(year) <= 5:
                        stats['recent_5_years'] += 1
                except ValueError:
                    pass
        
        return stats

# 예제 사용
formatter = ReferenceFormatter()

# 2025년 AI 정책 관련 핵심 문헌 추가
print("=== Reference Management System Demo ===\n")

# Journal articles
formatter.add_reference(
    'journal',
    authors=['Dafoe, A.', 'Zhang, B.', 'Anderljung, M.'],
    year='2025',
    title='Cooperative AI: Machines Must Learn to Find Common Ground',
    journal='Nature',
    volume='593',
    issue='7857',
    pages='33-36'
)

formatter.add_reference(
    'journal',
    authors=['Park, J.H.', 'Kim, S.Y.', 'Lee, M.J.'],
    year='2025',
    title='AI-Driven Policy Analysis in Korean Government: A Systematic Review',
    journal='Policy Sciences',
    volume='58',
    issue='2',
    pages='245-268'
)

# Government reports
formatter.add_reference(
    'government',
    organization='European Commission',
    year='2025',
    title='EU Artificial Intelligence Act: Implementation Guidelines',
    publisher='EU Publications Office',
    report_number='COM(2025)206'
)

formatter.add_reference(
    'government',
    organization='한국 과학기술정보통신부',
    year='2025',
    title='국가 AI 전략 2025-2030: 신뢰할 수 있는 AI 구현 로드맵',
    publisher='정부간행물',
    report_number='MSIT-2025-AI-001'
)

# Books
formatter.add_reference(
    'book',
    authors=['Russell, S.', 'Norvig, P.'],
    year='2025',
    title='Artificial Intelligence: A Modern Approach (5th Edition)',
    publisher='Pearson'
)

formatter.add_reference(
    'book',
    authors=['O\'Neil, C.'],
    year='2024',
    title='Weapons of Math Destruction: How Big Data Increases Inequality',
    publisher='Crown Publishing'
)

# Websites
formatter.add_reference(
    'website',
    author='OECD',
    year='2025',
    title='OECD AI Policy Observatory: Trends and Insights',
    url='https://oecd.ai/en/trends-and-data'
)

formatter.add_reference(
    'website',
    author='Partnership on AI',
    year='2025',
    title='Responsible AI Implementation Framework',
    url='https://partnershiponai.org/framework'
)

# Technical reports
formatter.add_reference(
    'government',
    organization='National Institute of Standards and Technology',
    year='2025',
    title='AI Risk Management Framework 2.0',
    publisher='NIST',
    report_number='NIST AI 100-2'
)

formatter.add_reference(
    'journal',
    authors=['Chen, X.', 'Wang, L.', 'Zhang, Y.'],
    year='2024',
    title='Causal AI for Public Policy: Methods and Applications',
    journal='Journal of Machine Learning Research',
    volume='25',
    issue='1',
    pages='1-42'
)

# 통계 분석
print("1. Reference Statistics:")
stats = formatter.analyze_references()
print(f"   Total references: {stats['total']}")
print(f"   Recent (2021-2025): {stats['recent_5_years']}")
print("\n   By type:")
for ref_type, count in stats['by_type'].items():
    print(f"     - {ref_type}: {count}")
print("\n   By year:")
for year in sorted(stats['by_year'].keys(), reverse=True):
    print(f"     - {year}: {stats['by_year'][year]}")

# APA 형식 출력
print("\n2. APA Format Output (Sample):")
apa_refs = formatter.format_all('apa')
for i, ref in enumerate(apa_refs[:3], 1):
    print(f"   {i}. {ref}")
print("   ...")

# BibTeX 형식 출력
print("\n3. BibTeX Format Output (Sample):")
bibtex_refs = formatter.format_all('bibtex')
print(bibtex_refs[0])

# 파일로 저장
formatter.save_to_file('references_apa.txt', 'apa')
formatter.save_to_file('references.bib', 'bibtex')

print("\n4. Files Generated:")
print("   - references_apa.txt (APA format)")
print("   - references.bib (BibTeX format)")

# 카테고리별 분류
print("\n5. Categorized References:")
categories = {
    'AI Ethics & Law': ['EU Artificial Intelligence Act', 'Responsible AI', 'AI Risk Management'],
    'Technical Methods': ['Causal AI', 'Cooperative AI', 'Artificial Intelligence: A Modern'],
    'Korean Context': ['Korean Government', '국가 AI 전략'],
    'Critical Analysis': ['Weapons of Math']
}

for category, keywords in categories.items():
    print(f"\n   {category}:")
    for ref in formatter.references:
        title = ref.get('title', '')
        if any(keyword in title for keyword in keywords):
            year = ref.get('year', 'n.d.')
            authors = ref.get('authors', [ref.get('author', ref.get('organization', 'Unknown'))])
            if isinstance(authors, list):
                author_str = authors[0].split(',')[0] if authors else 'Unknown'
            else:
                author_str = authors.split(',')[0] if authors else 'Unknown'
            print(f"     • {author_str} ({year}): {title[:60]}...")

print("\n=== Reference Management Complete ===")
print(f"Total: {stats['total']} references managed and formatted")
print(f"Recency: {stats['recent_5_years']}/{stats['total']} from last 5 years " 
      f"({100*stats['recent_5_years']/stats['total']:.0f}%)")