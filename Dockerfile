FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        python3 \
        python3-venv \
        python3-pip \
        && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

COPY requirements.txt .

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -U -r requirements.txt


FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS runtime

# Install system dependencies including ffmpeg and Python shared libraries
# Ubuntu 24.04 provides Python 3.12 and FFmpeg 6.x by default
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        python3 \
        python3-pip \
        libpython3.12 \
        python3.12-dev \
        && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /Whisper-WebUI

COPY . .
COPY --from=builder /Whisper-WebUI/venv /Whisper-WebUI/venv
# Backup venv to preserve it when source code is mounted as volume
COPY --from=builder /Whisper-WebUI/venv /venv-backup

# Copy and make entrypoint script executable
COPY docker-entrypoint.sh /Whisper-WebUI/docker-entrypoint.sh
RUN chmod +x /Whisper-WebUI/docker-entrypoint.sh

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]

ENV PATH="/Whisper-WebUI/venv/bin:$PATH"
# CUDA libraries (including NPP) are available in NVIDIA CUDA base image at /usr/local/cuda/lib64
# Add PyTorch CUDA libraries, Python shared libraries, and FFmpeg libraries to LD_LIBRARY_PATH
# Updated for Python 3.12 (Ubuntu 24.04 default)
ENV LD_LIBRARY_PATH=/Whisper-WebUI/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/Whisper-WebUI/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/Whisper-WebUI/venv/lib64/python3.12/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib64/python3.12/site-packages/nvidia/cudnn/lib:/Whisper-WebUI/venv/lib64/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/python3.12/config-3.12-x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Entrypoint will be overridden by docker-compose.yaml for development
# Default entrypoint preserved for production use
ENTRYPOINT [ "/Whisper-WebUI/docker-entrypoint.sh", "python", "app.py" ]
