# MapTrace Configuration System

This directory contains the Hydra-based configuration system for the MapTrace training pipeline. Configuration is organized into group files for composability and type-checked via dataclasses.

## Directory Structure

```
configs/
├── README.md                        # This file
├── __init__.py                      # Public API exports
├── base.yaml                        # Root config: defaults list + hydra block
├── config_schema.py                 # Dataclass schemas for type safety
│
├── paths/                           # File system paths
│   ├── paths.yaml                   # @package paths  — data/logs/weights roots
│   └── checkpoints.yaml             # @package checkpoint  — load paths
│
├── trainer/                         # Training hyperparameters
│   ├── trainer.yaml                 # @package trainer  — lr, batch, epochs, etc.
│   ├── loss_weights.yaml            # @package trainer.loss_weights
│   ├── patch_loss.yaml              # @package trainer.patch_loss
│   ├── adversarial.yaml             # @package trainer.adversarial
│   ├── ema.yaml                     # @package trainer.ema
│   └── lr_scheduler.yaml            # @package trainer.lr_scheduler
│
├── logs/                            # Monitoring / callback flags
│   └── logs.yaml                    # @package trainer.monitoring
│
├── lightning/                       # PyTorch Lightning Trainer settings
│   ├── lightning.yaml               # @package lightning
│   ├── callbacks.yaml               # @package lightning.callbacks
│   └── logger.yaml                  # @package lightning.logger
│
├── dataset/                         # Dataset definitions
│   ├── usgs_crops_512.yaml
│   ├── usgs_crops_512_trace_2.yaml
│   └── SpaceNet3.yaml
│
├── model/                           # Model architecture definitions
│   ├── deeplab_unet_precise.yaml
│   ├── deeplab_resnet50.yaml
│   ├── deeplab_resnet101.yaml
│   ├── resnet50_fpn.yaml
│   ├── resnet101_fpn.yaml
│   └── attention_unet.yaml
│
└── experiment/                      # Full experiment overrides
    ├── ema_base.yaml                # Default experiment (loaded by base.yaml)
    ├── smoke_test.yaml
    ├── smoke_test_2.yaml
    ├── smoke_test_3.yaml
    ├── smoke_test_4.yaml
    ├── ema_baseline_deeplab_unet_precise.yaml
    ├── small_ema_baseline_deeplab_unet_precise.yaml
    ├── extra_small_ema_baseline_deeplab_unet_precise.yaml
    ├── lr_0.0001_baseline_dl_unet_p_2.0_binary_roads.yaml
    └── lr_0.0001_int_baseline_dl_unet_p.yaml
```

## Quick Start

```bash
# Validate config composition without training
python train_lightning.py --cfg job experiment=smoke_test_4

# Run a smoke test
python train_lightning.py experiment=smoke_test_4 lightning.devices=[0]

# Full training run
python train_lightning.py experiment=ema_base lightning.devices=[0,1]
```

## Config Composition

`base.yaml` defines the full defaults list. Every group file uses a `# @package` directive that controls where it lands in `cfg`:

```yaml
defaults:
  - _self_
  - paths@paths: paths           # → cfg.paths
  - paths@checkpoint: checkpoints # → cfg.checkpoint  (two files, one group)
  - trainer: trainer             # → cfg.trainer
  - trainer/loss_weights: loss_weights   # → cfg.trainer.loss_weights
  - trainer/patch_loss: patch_loss       # → cfg.trainer.patch_loss
  - trainer/adversarial: adversarial     # → cfg.trainer.adversarial
  - trainer/ema: ema                     # → cfg.trainer.ema
  - trainer/lr_scheduler: lr_scheduler   # → cfg.trainer.lr_scheduler
  - logs: logs                           # → cfg.trainer.monitoring
  - lightning: lightning                 # → cfg.lightning
  - lightning/callbacks: callbacks       # → cfg.lightning.callbacks
  - lightning/logger: logger             # → cfg.lightning.logger
  - experiment: ema_base                 # → cfg.* (global overrides)
```

## CLI Overrides

Any config field is overridable at the CLI using dot-path syntax:

```bash
# Switch model and dataset
python train_lightning.py model=deeplab_resnet50 dataset=usgs_crops_512

# Tune optimizer
python train_lightning.py trainer.learning_rate=0.0001 trainer.batch_size=4

# Enable EMA
python train_lightning.py trainer.ema.use_ema=true trainer.ema.ema_start=10

# Adjust loss weights
python train_lightning.py trainer.loss_weights.roads=0.5 trainer.loss_weights.roads_patch=1.0

# Change devices
python train_lightning.py lightning.devices=[0,1] lightning.strategy=ddp

# Subset data for debugging
python train_lightning.py dataset.train_subset_size=500 dataset.val_subset_size=100
```

## Environment Variables

All path defaults resolve from environment variables with fallbacks:

| Variable             | Default              | Config field               |
|----------------------|----------------------|----------------------------|
| `DATA_ROOT`          | `/data2`             | `paths.data_root`          |
| `CHECKPOINTS_FOLDER` | `runs/checkpoints`   | `paths.checkpoints_folder` |
| `LOGS_FOLDER`        | `runs/logs`          | `paths.logs_folder`        |
| `WEIGHTS_FOLDER`     | `runs/weights`       | `paths.weights_folder`     |

## Schema Classes (`config_schema.py`)

| Class | Destination |
|---|---|
| `TopLevelConfig` | root `cfg` |
| `PathsConfig` | `cfg.paths` |
| `CheckpointConfig` | `cfg.checkpoint` |
| `DatasetConfig` | `cfg.dataset` |
| `ModelConfig` | `cfg.model` |
| `TrainerConfig` | `cfg.trainer` |
| `LossWeightsConfig` | `cfg.trainer.loss_weights` |
| `PatchLossConfig` | `cfg.trainer.patch_loss` |
| `AdversarialConfig` | `cfg.trainer.adversarial` |
| `EMAConfig` | `cfg.trainer.ema` |
| `LRSchedulerConfig` | `cfg.trainer.lr_scheduler` |
| `MonitoringConfig` | `cfg.trainer.monitoring` |
| `LightningTrainerConfig` | `cfg.lightning` |
| `LightningCallbacksConfig` | `cfg.lightning.callbacks` |
| `LightningLoggerConfig` | `cfg.lightning.logger` |

## Adding Experiments

Copy the closest existing experiment and edit:

```bash
cp configs/experiment/ema_base.yaml configs/experiment/my_experiment.yaml
# edit my_experiment.yaml
python train_lightning.py experiment=my_experiment
```

Experiment files use `# @package _global_` and override any field by path. They also
declare their own `defaults:` to select `model` and `dataset`:

```yaml
# @package _global_
defaults:
  - /model: deeplab_unet_precise
  - /dataset: usgs_crops_512_trace_2
  - _self_

name: "my_experiment"

trainer:
  learning_rate: 0.00001
  ema:
    use_ema: true
```

## Checkpoint Loading

To resume training or fine-tune from a checkpoint:

```bash
# Resume from Lightning checkpoint (restores optimizer state)
python train_lightning.py checkpoint.checkpoint_path=/path/to/checkpoint.ckpt

# Load weights only (no optimizer state)
python train_lightning.py checkpoint.weights_path=/path/to/weights.pth
```
