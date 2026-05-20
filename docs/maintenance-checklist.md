# Half-Release Maintenance Checklist

## Runtime Inputs

- `shadow_config.json`
- `data/cifar-10-batches-bin/data_batch_1.bin`
- `data/cifar-10-batches-bin/test_batch.bin`
- `python_server/CIFAR10/target/<cluster>/best_checkpoint_ep.pth`
- `python_server/CIFAR10/shadow_json_aligned/best_checkpoint_ep.pth`

## Docker Services

- `target-oracle`: target model FastAPI service
- `shadow-oracle`: shadow model FastAPI service
- `audit-runner`: Go audit entrypoint
- `threshold-generator`: optional tool service for regenerating `shadow_config.json`

## Key Environment Variables

- `TARGET_MODEL_PATH`
- `SHADOW_MODEL_PATH`
- `LABELSCAN_PRESET`
- `SHADOW_JSON_PATH`
- `SHADOW_TRAIN_INDICES_PATH`
- `SHADOW_RED_QUANTILE`

## Before Releasing

- Confirm both model checkpoints exist at the configured paths.
- Confirm `shadow_config.json` matches the shadow model in use.
- Confirm `data/cifar-10-batches-bin/` contains the binary CIFAR files needed by Go.
- Run the Dockerized audit flow and verify `output/audit_report.json` and `output/audit_report.html` are updated.
- If the shadow model changes, rerun `threshold-generator` before packaging.

## High-Risk Points

- Go uses binary CIFAR files under `data/`, while threshold regeneration uses the real `shadow_train_data.json` file at the repository root.
- The Go runtime depends on both target and shadow HTTP services being healthy before it starts.
- If `shadow_train_data.json` is absent, audit can still run with the existing `shadow_config.json`, but thresholds cannot be regenerated.
