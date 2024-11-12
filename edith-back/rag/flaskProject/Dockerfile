## 베이스 이미지 선택
#FROM python:3.10-slim
#
## 빌드에 필요한 기본 도구들 설치
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    git \
#    && rm -rf /var/lib/apt/lists/*
#
## 작업 디렉터리 설정
#WORKDIR /app
#
## requirements.txt 먼저 복사
#COPY requirements.txt .
#
## pip 설치 최적화
#RUN pip install --no-cache-dir --upgrade pip && \
#    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
#    pip install --no-cache-dir transformers chromadb langchain openai langchain-community \
#    langchain-openai python-dotenv sentence-transformers && \
#    pip install --no-cache-dir -r requirements.txt
#
## tree-sitter 관련 설치
#RUN pip install --no-cache-dir tree-sitter tree-sitter-python tree-sitter-javascript \
#    tree-sitter-java tree-sitter-c tree-sitter-cpp python-gitlab gitpython
#
## 로그 디렉토리 생성 및 권한 설정
#RUN mkdir -p /app/logs && \
#    chmod 777 /app/logs
#
## 나머지 파일 복사
#COPY . .
#
## Flask 환경 변수 설정
#ENV FLASK_APP=run.py
#ENV FLASK_RUN_PORT=8083
#
## gunicorn 실행
#CMD ["gunicorn", "--bind", "0.0.0.0:8083", "--workers=4", "--timeout=180", "--log-level=info", "--error-logfile=/app/logs/error.log", "--access-logfile=/app/logs/access.log", "run:app"]

# 베이스 이미지 선택
FROM python:3.10-slim

# 빌드에 필요한 기본 도구들 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# requirements.txt 복사 (캐시 사용을 위한 배치 순서)
COPY requirements.txt .

# 패키지 설치 (캐시를 최대한 활용하기 위해 requirements.txt 먼저 설치)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers chromadb langchain openai langchain-community \
    langchain-openai python-dotenv sentence-transformers && \
    pip install -r requirements.txt

# 나머지 파일 복사 (자주 변경될 가능성이 있는 파일)
COPY . .

# 로그 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Flask 환경 변수 설정
ENV FLASK_APP=run.py
ENV FLASK_RUN_PORT=8083

# gunicorn 실행
CMD ["gunicorn", "--bind", "0.0.0.0:8083", "--workers=4", "--timeout=180", "--log-level=info", "--error-logfile=/app/logs/error.log", "--access-logfile=/app/logs/access.log", "run:app"]
