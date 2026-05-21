"""
Configuration dataclasses for GEODiffusion training system.

Provides type hints and IDE support for the Hydra config tree.
The actual values are loaded from YAML files under configs/.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from omegaconf import MISSING


@dataclass
class PathsConfig:
    data_root: str = "${oc.env:DATA_ROOT,/data2}"
    checkpoints_folder: str = "${oc.env:CHECKPOINTS_FOLDER,runs/checkpoints}"
    logs_folder: str = "${oc.env:LOGS_FOLDER,runs/logs}"
    weights_folder: str = "${oc.env:WEIGHTS_FOLDER,runs/weights}"
    hydra_outputs: str = "${oc.env:HYDRA_OUTPUTS,runs/hydra}"


@dataclass
class CheckpointConfig:
    checkpoint_path: Optional[str] = None
    weights_path: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str = MISSING
    data_root: str = MISSING
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None


@dataclass
class ModelConfig:
    name: str = "transformer"
    max_segments: int = 500
    img_feature_dim: int = 512


@dataclass
class LrSchedulerConfig:
    type: str = "cosine"
    params: Any = field(default_factory=dict)


@dataclass
class TrainerConfig:
    timesteps: int = 200
    schedule_type: str = "cosine"
    eval_timesteps: List[int] = field(default_factory=lambda: [0, 25, 50, 100, 150, 199])
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 200
    batch_size: int = 4
    num_workers: int = 4
    gradient_clipping: float = 1.0
    lambda_mask: float = 1.0
    lambda_endpoint: float = 0.0
    snap_endpoints: bool = True
    snap_threshold: float = 0.02
    snap_mode: str = "self"
    lr_scheduler: Optional[LrSchedulerConfig] = None


@dataclass
class LightningCallbacksConfig:
    checkpoint_monitor: str = "Loss/val"
    checkpoint_mode: str = "min"
    checkpoint_save_top_k: int = 1
    checkpoint_save_last: bool = True
    checkpoint_every_n_epochs: int = 5


@dataclass
class LightningLoggerConfig:
    tensorboard_log_dir: str = "${paths.logs_folder}"
    csv_log_dir: str = "${paths.logs_folder}"


@dataclass
class LightningConfig:
    accelerator: str = "gpu"
    devices: Any = "1"
    num_nodes: int = 1
    strategy: Optional[str] = None
    precision: str = "bf16-mixed"
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50
    callbacks: LightningCallbacksConfig = field(default_factory=LightningCallbacksConfig)
    logger: LightningLoggerConfig = field(default_factory=LightningLoggerConfig)


@dataclass
class TopLevelConfig:
    name: str = "UNNAMED"
    experiment_run_id: str = "${name}"
    val_image_log_every_n_epochs: int = 10
    paths: PathsConfig = field(default_factory=PathsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    lightning: LightningConfig = field(default_factory=LightningConfig)
    channels: int = 3
    normalize: bool = True
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    use_augmentation: bool = True
    augmentation: Optional[Dict[str, Any]] = None
    filter_nir: bool = False
    valid: bool = True
    skip_empty_valid_masks: bool = False
    filter_nodata: bool = False
    nodata_threshold: float = 0.30
    filter_density_extremes: bool = False
    output_mode: str = "roads_and_crossings"
    return_type_map: bool = False
    # Optional list of additional dataset roots to concatenate with data_root.
    # Each entry is a path string; all sub-datasets share the same settings.
    extra_data_roots: Optional[List[str]] = None
    # Which splits to load from each extra root. Defaults to ["train"].
    # Set to ["train", "val"] to absorb the validation split of extra datasets
    # into training (valid because evaluation uses the primary root's val set).
    extra_data_splits: Optional[List[str]] = None

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = MISSING  # e.g., "deeplab_unet_precise", "deeplabv3_resnet101"
    num_outputs: int = MISSING  # Number of output channels
    outputs: Dict[str, int] = field(default_factory=dict)  # Output channel mapping
    in_channels: int = 3
    variant: Optional[str] = None
    pretrained: bool = True

@dataclass
class PatchLossConfig:
    """Patch alignment loss configuration."""
    patch_size: int = 3
    patch_reciprocal: bool = False
    recip_max_dist: int = 5
    recip_temp: float = 0.1
    recip_alpha: float = 5.0
    recip_normalization: str = "max"
    recip_singleton_penalty: float = 0.5
    # Intersection loss
    intersection_loss_weight: float = 0.0
    # Focal loss
    patch_use_focal_loss: bool = False
    patch_focal_alpha: float = 0.25
    patch_focal_gamma: float = 2.0

@dataclass
class AdversarialConfig:
    """Adversarial training configuration."""
    critic_learning_rate: float = 0.0001
    critic_type: str = "roads"  # 'roads' or 'dist'
    critic_channels: Optional[List[int]] = None
    adversarial_loss_weight: float = 0.1
    critic_update_period: int = 1
    critic_weight_decay: float = 0.001

@dataclass
class EMAConfig:
    """Exponential Moving Average configuration."""
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_eval_freq: int = 1
    ema_start: int = 10  # Epoch/step when EMA starts
    num_outliers_percentile: int = 95  # Percentile used for threshold
    outlier_weight: float = 0.1  # Weight for "bad" samples
    gamma: float = 0.95  # EMA loss blending factor

@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "none"  # 'none', 'cosine', 'step', 'plateau'
    params: Dict = field(default_factory=dict)

@dataclass
class MonitoringConfig:
    """Monitoring callback configuration."""
    log_gpu_metrics: bool = False
    log_gradient_stats: bool = False
    log_training_monitor: bool = False
    log_progressive_viz: bool = True
    log_random_val_grid: bool = True
    log_extended_val_metrics: bool = True
    log_ema_scalars: bool = True
    log_prediction_diagnostics: bool = False
    log_per_step: bool = False
    log_per_layer_gradients: bool = False
    log_model_stats: bool = False
    viz_num_samples: int = 20
    viz_log_every_n_epochs: int = 1

@dataclass
class TrainerConfig:
    """Training hyperparameters and optimization configuration."""
    # Optimization
    learning_rate: float = 0.0001
    batch_size: int = 16
    num_epochs: int = 100
    weight_decay: float = 0.00001
    gradient_clipping: Optional[float] = None
    num_workers: int = 4

    # Loss configuration
    pos_weight: float = 50.0  # Positive weight for imbalanced classes
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)

    # Patch loss
    patch_loss: PatchLossConfig = field(default_factory=PatchLossConfig)

    # Adversarial training
    use_adversarial_training: bool = False
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)

    # EMA
    ema: EMAConfig = field(default_factory=EMAConfig)

    # Learning rate scheduling
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Monitoring callbacks
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Dilated (buffer-tolerant) validation metrics
    dilated_metric_radius: int = 0  # 0 = disabled; 3 = 7x7 kernel (3px buffer each side)

@dataclass
class LightningCallbacksConfig:
    """PyTorch Lightning callbacks configuration."""
    # Model checkpointing
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    checkpoint_save_top_k: int = 3
    checkpoint_save_last: bool = True
    
    # Early stopping
    early_stop_monitor: str = "val_loss"  
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    early_stop_mode: str = "min"
    
    # Checkpoint every N epochs
    checkpoint_every_n_epochs: int = 5  # Save checkpoints every N epochs

@dataclass
class LightningLoggerConfig:
    """PyTorch Lightning logger configuration."""
    # TensorBoard logging
    tensorboard_log_dir: str = "${paths.logs_folder}"
    csv_log_dir: str = "${paths.logs_folder}"

@dataclass
class LightningTrainerConfig:
    """PyTorch Lightning Trainer configuration."""
    # GPU/device configuration
    accelerator: str = "gpu"  # 'gpu', 'cpu', 'tpu'
    devices: Any = "auto"  # 'auto', list of device IDs, or number of devices
    num_nodes: int = 1  # Number of nodes for multi-node training
    strategy: Optional[str] = None  # 'ddp', 'dp', etc.
    
    # Precision
    precision: str = "16-mixed"  # '16-mixed', '32', 'bf16-mixed'
    
    # Validation and logging
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50
    
    # Callbacks and loggers (will be configured programmatically)
    callbacks: LightningCallbacksConfig = field(default_factory=LightningCallbacksConfig)
    logger: LightningLoggerConfig = field(default_factory=LightningLoggerConfig)

@dataclass
class TopLevelConfig:
    """Top-level training configuration combining all components."""
    # Paths and checkpointing
    paths: PathsConfig = field(default_factory=PathsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Core components (can be composed via Hydra)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # Lightning trainer configuration
    lightning: LightningTrainerConfig = field(default_factory=LightningTrainerConfig)

    # Experiment naming
    name: str = MISSING
    val_image_log_every_n_epochs: int = 1
    val_image_log_max_items: int = 20

    # Experiment run identifier
    experiment_run_id: str = "default_experiment"

