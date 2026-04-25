# Half-Release Maintenance Checklist

## Runtime Inputs

- `shadow_config.json`
- `data/cifar-10-batches-bin/data_batch_1.bin`
- `data/cifar-10-batches-bin/test_batch.bin`
- `python_server/CIFAR10/target/<cluster>/best_checkpoint_ep.pth`
- `python_server/CIFAR10/shadow/<cluster>/best_checkpoint_ep.pth`

## Docker Services

- `target-oracle`: target model FastAPI service
- `shadow-oracle`: shadow model FastAPI service
- `audit-runner`: Go audit entrypoint
- `threshold-generator`: optional tool service for regenerating `shadow_config.json`

## Key Environment Variables

- `TARGET_MODEL_PATH`
- `SHADOW_MODEL_PATH`
- `TARGET_API`
- `SHADOW_API`
- `SHADOW_CONFIG_PATH`
- `CALIBRATION_DATA_PATH`
- `MEMBER_DATA_PATH`
- `NON_MEMBER_DATA_PATH`
- `OUTPUT_REPORT_PATH`

## Before Releasing

- Confirm both model checkpoints exist at the configured paths.
- Confirm `shadow_config.json` matches the shadow model in use.
- Confirm `data/cifar-10-batches-bin/` contains the binary CIFAR files needed by Go.
- Run the Dockerized audit flow and verify `output/audit_report.json` is updated.
- If the shadow model changes, rerun `threshold-generator` before packaging.

## High-Risk Points

- Go uses binary CIFAR files under `data/`, while Python threshold generation uses `python_server/data/`.
- The Go runtime depends on both target and shadow HTTP services being healthy before it starts.
- Threshold fields are normalized at runtime for backward compatibility; keep `tau_95` and `tau_opt` when regenerating configs.
- If you change service names or ports in Compose, update `TARGET_API` and `SHADOW_API` together.
