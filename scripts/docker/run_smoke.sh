#!/usr/bin/env bash
set -euo pipefail

LABELSCAN_PRESET=smoke \
docker compose up --build --abort-on-container-exit --exit-code-from audit-runner audit-runner
