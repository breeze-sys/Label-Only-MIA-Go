#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist/labelscan-halfrelease"

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

copy_into_dist() {
  local path="$1"
  local target="${DIST_DIR}/${path}"
  mkdir -p "$(dirname "${target}")"
  cp -a "${ROOT_DIR}/${path}" "${target}"
}

copy_into_dist "README.md"
copy_into_dist ".env.example"
copy_into_dist ".dockerignore"
copy_into_dist ".gitignore"
copy_into_dist "docker-compose.yml"
copy_into_dist "main.go"
copy_into_dist "go.mod"
copy_into_dist "pkg"
copy_into_dist "docker"
copy_into_dist "docs"
copy_into_dist "scripts/docker"
copy_into_dist "python_server/server.py"
copy_into_dist "python_server/classifier.py"
copy_into_dist "python_server/utils.py"
copy_into_dist "python_server/calc_thresholds.py"
copy_into_dist "python_server/requirements.txt"
copy_into_dist "python_server/CIFAR10"
copy_into_dist "python_server/data"
copy_into_dist "data/cifar-10-batches-bin"
copy_into_dist "shadow_config.json"

if [ -d "${ROOT_DIR}/output" ]; then
  copy_into_dist "output"
fi

cat > "${DIST_DIR}/RELEASE_NOTES.txt" <<'EOF'
LabelScan half-release bundle

Recommended first command:
  ./scripts/docker/run_smoke.sh

Full audit command:
  ./scripts/docker/run_halfrelease.sh

Shadow threshold regeneration:
  ./scripts/docker/rebuild_shadow_config.sh

Competition-oriented instructions:
  docs/competition-delivery.md
EOF

echo "Prepared release bundle at: ${DIST_DIR}"
