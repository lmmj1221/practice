"""
경로 수정 표준 템플릿

모든 Python 파일 상단에 추가:

from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
data_dir = current_dir.parent / 'data'
output_dir = current_dir.parent / 'outputs'

# Create directories if needed
data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

사용 예시:
- CSV 읽기: pd.read_csv(data_dir / 'filename.csv')
- CSV 저장: df.to_csv(data_dir / 'filename.csv')
- PNG 저장: plt.savefig(output_dir / 'filename.png')
"""
