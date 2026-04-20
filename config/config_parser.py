import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class DatasetConfig:
    mode: str
    root: str
    download: bool


@dataclass
class ModelConfig:
    name: str
    dropout: float
    pretrained: bool
    checkpoint_dir: str = './checkpoints'  
    load_pretrained: bool = True  


@dataclass
class FederatedConfig:
    n_nodes: int
    n_rounds: int
    samples_per_client: int
    min_samples: int
    max_samples: int
    local_epochs: int
    fail_safe_alpha: float

@dataclass
class TrainingConfig:
    batch_size: int
    eval_batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float

@dataclass
class AttackConfig:
    enabled: bool
    malicious_clients: int
    target_label: int
    types: List[str]

@dataclass
class EncryptionConfig:
    enabled: bool
    type: str
    poly_modulus_degree: int
    coeff_mod_bit_sizes: tuple
    global_scale: int

@dataclass
class AggregationConfig:
    mode: str
    he_enabled: Optional[bool]
    selection_strategy: str

@dataclass
class DeviceConfig:
    type: str

@dataclass
class LoggingConfig:
    level: str
    verbose: bool
    save_logs: bool
    log_file: str

@dataclass
class CheckpointConfig:
    enabled: bool
    save_dir: str
    save_interval: int

@dataclass
class PhasesConfig:
    stable_initialization: int
    gradual_attack_introduction: int
    moderate_attack_frequency: int
    high_intensity_attacks: int
    consecutive_attack_stress_test: int
    recovery_and_final_validation: int

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    federated: FederatedConfig
    training: TrainingConfig
    attack: AttackConfig
    encryption: EncryptionConfig
    aggregation: AggregationConfig
    device: DeviceConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    phases: PhasesConfig
    seed: int

    def update_he_flag(self):
        """synchronize HE flags from aggregation or encryption config"""
        if self.aggregation.he_enabled is not None:
            self.encryption.enabled = self.aggregation.he_enabled
        self.aggregation.he_enabled = self.encryption.enabled


class ConfigParser:
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """load YAML configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def parse(config_path: Optional[str] = None, **kwargs) -> Config:
        """parse configuration from YAML and CLI arguments"""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        # load from YAML
        config_dict = ConfigParser.load_yaml(str(config_path))

        # 0verride with CLI arguments
        ConfigParser._update_nested_dict(config_dict, kwargs)

        # create dataclass instances
        config = Config(
            dataset=DatasetConfig(**config_dict['dataset']),
            model=ModelConfig(**config_dict['model']),
            federated=FederatedConfig(**config_dict['federated']),
            training=TrainingConfig(**config_dict['training']),
            attack=AttackConfig(**config_dict['attack']),
            encryption=EncryptionConfig(
                enabled=config_dict['encryption']['enabled'],
                type=config_dict['encryption']['type'],
                poly_modulus_degree=config_dict['encryption']['poly_modulus_degree'],
                coeff_mod_bit_sizes=tuple(config_dict['encryption']['coeff_mod_bit_sizes']),
                global_scale=config_dict['encryption']['global_scale']
            ),
            aggregation=AggregationConfig(**config_dict['aggregation']),
            device=DeviceConfig(**config_dict['device']),
            logging=LoggingConfig(**config_dict['logging']),
            checkpoint=CheckpointConfig(**config_dict['checkpoint']),
            phases=PhasesConfig(**config_dict['phases']),
            seed=config_dict['seed']
        )

        # synchronize HE flags
        config.update_he_flag()

        return config

    @staticmethod
    def _update_nested_dict(d: Dict, updates: Dict):
        """update nested dictionaries"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                ConfigParser._update_nested_dict(d[key], value)
            elif value is not None:
                d[key] = value


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='FedBagSafe with Optional Homomorphic Encryption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
command samples:
  python main.py --config config/config.yaml
  python main.py --dataset fashionmnist --enable-he
  python main.py --dataset cifar10 --disable-he --n-rounds 50
  python main.py --dataset femnist --enable-he --n-nodes 10 or ...
        """
    )

    # configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')

    # dataset
    parser.add_argument('--dataset', type=str,
                       choices=['cifar10', 'cifar100', 'fashionmnist', 'femnist'],
                       help='Dataset to use')

    # model
    parser.add_argument('--model', type=str,
                       choices=['resnet9', 'cifar_resnet'],
                       help='Model architecture')

    # federated Learning
    parser.add_argument('--n-nodes', type=int, help='Number of clients')
    parser.add_argument('--n-rounds', type=int, help='Number of federated rounds')
    parser.add_argument('--malicious-clients', type=int,
                       help='Number of malicious clients')

    # training
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Local training epochs')

    # encrypton
    parser.add_argument('--enable-he', action='store_true',
                       help='Enable Homomorphic Encryption')
    parser.add_argument('--disable-he', action='store_true',
                       help='Disable Homomorphic Encryption')

    # attacks
    parser.add_argument('--disable-attacks', action='store_true',
                       help='Disable all attacks')
    parser.add_argument('--enable-attacks', action='store_true',
                       help='Enable attacks')

    #device
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='Device to use (cpu or cuda)')

    # logging
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--log-level', type=str,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    # validation
    parser.add_argument('--mode', type=str,
                       choices=['deterministic', 'random'],
                       help='Client selection mode')

    return parser


def process_cli_args(args) -> Dict[str, Any]:
    """convert CLI arguments to config update dictionary"""
    updates = {}

    if args.dataset:
        updates['dataset'] = {'mode': args.dataset}

    if args.model:
        updates['model'] = {'name': args.model}

    if args.n_nodes or args.n_rounds or args.malicious_clients:
        updates['federated'] = {}
        if args.n_nodes:
            updates['federated']['n_nodes'] = args.n_nodes
        if args.n_rounds:
            updates['federated']['n_rounds'] = args.n_rounds
        if args.malicious_clients:
            updates['federated']['malicious_clients'] = args.malicious_clients

    if args.batch_size or args.lr or args.epochs:
        updates['training'] = {}
        if args.batch_size:
            updates['training']['batch_size'] = args.batch_size
        if args.lr:
            updates['training']['learning_rate'] = args.lr
        if args.epochs:
            updates['training']['local_epochs'] = args.epochs

    if args.enable_he or args.disable_he:
        updates['encryption'] = {'enabled': args.enable_he}
        updates['aggregation'] = {'he_enabled': args.enable_he}

    if args.disable_attacks or args.enable_attacks:
        updates['attack'] = {'enabled': args.enable_attacks}

    if args.device:
        updates['device'] = {'type': args.device}

    if args.seed:
        updates['seed'] = args.seed

    if args.log_level:
        updates['logging'] = {'level': args.log_level}

    if args.verbose:
        updates['logging'] = updates.get('logging', {})
        updates['logging']['verbose'] = True

    if args.mode:
        updates['aggregation'] = updates.get('aggregation', {})
        updates['aggregation']['selection_strategy'] = args.mode

    return updates