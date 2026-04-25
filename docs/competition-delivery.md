# Competition Delivery Guide

## Purpose

This project is packaged as a Docker-based half-release bundle for competition review. The intended review flow is:

1. Run a quick smoke test to confirm the environment, containers, model loading, and report generation all work.
2. Review the generated report and the precomputed full-run results or metrics.
3. Run the full audit only if the judges want a complete live execution and have enough time.

## Recommended Judge Workflow

### Fast Demo

Run:

```bash
./scripts/docker/run_smoke.sh
```

What it proves:

- Docker images build successfully
- Target and shadow model services start successfully
- Pretrained checkpoints load successfully
- The Go audit pipeline can connect to both services
- The system can generate `output/audit_report.json`

Typical result:

- Uses minimal calibration and audit sample counts
- Finishes much faster than the full audit

### Full Audit

Run:

```bash
./scripts/docker/run_halfrelease.sh
```

What it does:

- Uses the default runtime parameters from `docker-compose.yml` and `.env`
- Runs the real audit flow with more calibration and audit work

Important note:

- The model is pretrained, so there is no training during evaluation.
- The runtime is still long because the audit itself repeatedly performs black-box boundary attack queries.
- The slow part is the attack workload, not model training.

## Files Judges Need

- `docker-compose.yml`
- `docker/`
- `scripts/docker/`
- `docs/`
- `main.go`
- `go.mod`
- `pkg/`
- `python_server/`
- `data/cifar-10-batches-bin/`
- `shadow_config.json`
- `output/`

## What To Say During Delivery

- The smoke test is the recommended live demonstration path.
- The full audit is available, but it is intentionally slower because it performs repeated attack queries against the pretrained models.
- If the shadow model changes, `shadow_config.json` must be regenerated before release.
