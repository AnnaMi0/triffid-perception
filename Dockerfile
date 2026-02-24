# =============================================================
# TRIFFID Perception – Docker Image
# Base: ROS2 Humble on Ubuntu 22.04 with CUDA support
# =============================================================
FROM ros:humble-perception-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=0

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (inside container, safe) ─────────────────────
# Pin numpy<2 because ROS2 Humble cv_bridge was compiled against NumPy 1.x
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    ultralytics

# ── Workspace setup ──────────────────────────────────────────
WORKDIR /ws
# src/ is bind-mounted at runtime via docker-compose

# ── Entrypoint ───────────────────────────────────────────────
COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh
ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["bash"]
