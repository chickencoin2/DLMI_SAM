# DLMI-SAM Labeler
# GPU: pip cu128 wheels bundle CUDA runtime; host needs NVIDIA driver + nvidia-container-toolkit.
#   docker build -t dlmi-sam .
#   xhost +local:docker
#   docker run --rm -it --gpus all -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       dlmi-sam
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# git: pip VCS installs (transformers/sam3 pinned to git refs)
# libtk8.6/tcl8.6: tkinter runtime libs (python slim ships _tkinter without them)
# libgl1/libglib2.0-0: opencv-python runtime libs
# xvfb/x11-utils: headless GUI smoke test
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libtk8.6 tcl8.6 \
        libgl1 libglib2.0-0 \
        xvfb x11-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
