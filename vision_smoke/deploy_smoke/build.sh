#!/bin/sh
set -eu

cd "$(dirname "$0")"

IMAGE="vision-runtime:smoke"
OUTPUT="../vision-runtime-arm64.tar"

if command -v docker >/dev/null 2>&1 \
    && docker info >/dev/null 2>&1 \
    && docker buildx version >/dev/null 2>&1; then
    docker buildx build \
        --platform linux/arm64 \
        --tag "$IMAGE" \
        --output "type=docker,dest=$OUTPUT" \
        .
else
    echo "Docker buildx is unavailable; using the daemonless registry builder."
    python3 build_no_daemon.py --output "$OUTPUT"
fi

python3 inspect_archive.py "$OUTPUT"
ls -lh "$OUTPUT"
sha256sum "$OUTPUT"
