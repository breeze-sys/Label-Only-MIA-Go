#!/usr/bin/env bash
set -euo pipefail

CALIBRATION_CANDIDATE_COUNT=10 \
CALIBRATION_TARGET_COUNT=1 \
MIN_VALID_STRANGERS=1 \
MEMBER_SAMPLE_COUNT=1 \
NON_MEMBER_SAMPLE_COUNT=1 \
AUDIT_WORKERS=2 \
docker compose up --build --abort-on-container-exit --exit-code-from audit-runner audit-runner
