FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libffi-dev \
    libnacl-dev \
    python3 \
    python3-pip \
    nvidia-cuda-dev \
    cuda-nvrtc-12-1 \
    cuda-nvrtc-dev-12-1 \
    build-essential \
    curl \
    git \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

    # Pythonのシンボリックリンクを作成（pythonコマンドを使用可能にする）
RUN ln -s /usr/bin/python3 /usr/bin/python

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# タイムゾーンを設定
RUN ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && echo "Asia/Tokyo" > /etc/timezone
# アプリケーションのコピー

EXPOSE 8501

# Set environment variables for Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "appver2.py", "--server.port", "8501", "--server.address", "0.0.0.0"]