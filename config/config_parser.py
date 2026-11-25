import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: UP035

import yaml


@dataclass
class DatasetConfig:
    mode: str
    root: str
    data_dir: str
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
    min_samples: Optional[int]
    max_samples: Optional[int]
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

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)


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
    accelerator: str = 'auto'
    devices: Any = 'auto'


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
        """sync he flags between aggregation and encryption"""
        if self.aggregation.he_enabled is not None:
            self.encryption.enabled = self.aggregation.he_enabled
        self.aggregation.he_enabled = self.encryption.enabled


class ConfigParser:
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """load yaml config file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def parse(config_path: Optional[str] = None, **kwargs) -> Config:
        """parse config from yaml and cli args"""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        # load yaml
        config_dict = ConfigParser.load_yaml(str(config_path))

        # override with cli args
        ConfigParser._update_nested_dict(config_dict, kwargs)

        # build config objects
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

        # sync he flags
        config.update_he_flag()

        return config

    @staticmethod
    def _update_nested_dict(d: Dict, updates: Dict):
        """update nested dict values"""
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
examples:
  python main.py --config config/config.yaml
  python main.py --dataset fashionmnist --enable-he
  python main.py --dataset cifar10 --disable-he --n-rounds 50
  python main.py --dataset femnist --enable-he --n-nodes 10
        """
    )

    # config file
    parser.add_argument('--config', type=str, default=None,
                        help='path to yaml config file')

    # dataset
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'cifar100', 'fashionmnist', 'femnist'],
                        help='dataset to use')

    # model
    parser.add_argument('--model', type=str,
                        choices=['resnet9', 'cifar_resnet'],
                        help='model architecture')

    # federated learning
    parser.add_argument('--n-nodes', type=int, help='number of clients')
    parser.add_argument('--n-rounds', type=int, help='number of federated rounds')
    parser.add_argument('--malicious-clients', type=int,
                        help='number of malicious clients')

    # training
    parser.add_argument('--batch-size', type=int, help='training batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='local training epochs')

    # encryption
    parser.add_argument('--enable-he', action='store_true',
                        help='enable homomorphic encryption')
    parser.add_argument('--disable-he', action='store_true',
                        help='disable homomorphic encryption')

    # attacks
    parser.add_argument('--disable-attacks', action='store_true',
                        help='disable all attacks')
    parser.add_argument('--enable-attacks', action='store_true',
                        help='enable attacks')

    # device
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        help='device to use (cpu or cuda)')

    # logging
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose logging')
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='logging level')

    return parser


def process_cli_args(args) -> Dict[str, Any]:
    """convert cli args to config dict"""
    updates = {}

    if args.dataset:
        updates['dataset'] = {'mode': args.dataset}

    if args.model:
        updates['model'] = {'name': args.model}

    if args.n_nodes or args.n_rounds or args.epochs:
        updates.setdefault('federated', {})
        if args.n_nodes:
            updates['federated']['n_nodes'] = args.n_nodes
        if args.n_rounds:
            updates['federated']['n_rounds'] = args.n_rounds
        if args.epochs:
            updates['federated']['local_epochs'] = args.epochs

    if args.batch_size or args.lr:
        updates.setdefault('training', {})
        if args.batch_size:
            updates['training']['batch_size'] = args.batch_size
        if args.lr:
            updates['training']['learning_rate'] = args.lr

    if args.enable_he or args.disable_he:
        updates['encryption'] = {'enabled': args.enable_he}
        updates.setdefault('aggregation', {})
        updates['aggregation']['he_enabled'] = args.enable_he

    if args.disable_attacks or args.enable_attacks or args.malicious_clients is not None:
        updates.setdefault('attack', {})
        if args.disable_attacks or args.enable_attacks:
            updates['attack']['enabled'] = args.enable_attacks
        if args.malicious_clients is not None:
            updates['attack']['malicious_clients'] = args.malicious_clients

    if args.device:
        updates['device'] = {
            'type': args.device,
            'accelerator': args.device
        }

    if args.seed is not None:
        updates['seed'] = args.seed

    if args.log_level:
        updates['logging'] = {'level': args.log_level}

    if args.verbose:
        updates['logging'] = updates.get('logging', {})
        updates['logging']['verbose'] = True

    return updates