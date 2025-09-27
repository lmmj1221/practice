#!/bin/bash

# macOS용 가상환경 활성화 스크립트

echo "🐍 Python 가상환경 설정 중..."

# 가상환경이 없으면 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
    echo "✅ 가상환경 생성 완료"
else
    echo "✅ 기존 가상환경 발견"
fi

# 가상환경 활성화
source venv/bin/activate

echo "🚀 가상환경 활성화 완료!"
echo "📍 Python 경로: $(which python)"
echo "📌 Python 버전: $(python --version)"

# 패키지 업그레이드
echo ""
echo "📦 pip 업그레이드 중..."
pip install --upgrade pip --quiet

# requirements.txt가 있으면 패키지 설치
if [ -f "requirements.txt" ]; then
    echo "📋 requirements.txt 발견"
    echo "📦 패키지 설치 중..."
    pip install -r requirements.txt --quiet
    echo "✅ 패키지 설치 완료"
else
    echo "⚠️  requirements.txt 파일이 없습니다"
fi

echo ""
echo "🎉 환경 설정 완료!"
echo "💡 가상환경 비활성화: deactivate"
echo "💡 프로그램 실행: python education.py 또는 python analysis.py"