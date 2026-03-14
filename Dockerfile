# IsaacLab + overlay training image for lift-and-reach-RL-training-robot
#
# Assumptions:
# 1. The lab server can pull NVIDIA NGC images.
# 2. Training is headless inside the container.
# 3. This repo is an overlay on top of IsaacLab, not a standalone project.
#
# Build example:
# docker build -t isaaclab-lift:latest \
#   --build-arg USER_ID=$(id -u) \
#   --build-arg GROUP_ID=$(id -g) .
#
# Run example:
# docker run --rm -it --gpus '"device=0"' --shm-size 16G \
#   -v /path/on/server/logs:/workspace/logs \
#   --name isaaclab-lift isaaclab-lift:latest bash

FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1 \
    ISAACSIM_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/IsaacLab \
    OVERLAY_PATH=/workspace/lift-and-reach-RL-training-robot

USER root

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    curl \
    wget \
    tmux \
    htop \
    rsync \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxfixes3 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxxf86vm1 \
    libegl1 \
    libgl1 \
    libopengl0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG ISAACLAB_REF=main
RUN git clone --depth 1 --branch ${ISAACLAB_REF} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

WORKDIR ${ISAACLAB_PATH}
RUN ${ISAACSIM_PATH}/python.sh -m pip install \
    -e source/isaaclab \
    -e source/isaaclab_assets \
    -e source/isaaclab_tasks \
    -e source/isaaclab_mimic \
    -e source/isaaclab_rl
RUN ${ISAACSIM_PATH}/python.sh -m pip install \
    rsl-rl-lib==3.0.1 \
    tensordict \
    gymnasium
RUN ${ISAACSIM_PATH}/python.sh -m pip install --no-deps torchvision

WORKDIR /workspace
COPY . ${OVERLAY_PATH}

WORKDIR ${OVERLAY_PATH}
RUN ${ISAACSIM_PATH}/python.sh scripts/install_overlay.py \
    --task-root /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift \
    --rsl-root /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/rsl_rl

# Convenience launch script.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'cd /workspace/IsaacLab' \
    '${ISAACSIM_PATH}/python.sh scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-DualArm-IK-Rel-v0 --enable_cameras --headless --device cuda:0 --rendering_mode performance "$@"' \
    > /usr/local/bin/train_lift.sh && chmod +x /usr/local/bin/train_lift.sh

WORKDIR /workspace/IsaacLab

CMD ["/bin/bash"]
