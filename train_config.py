from dataclasses import dataclass


@dataclass
class TrainArgs:
    batch_size: int = 8
    max_steps: int = 30000 
    lr: float = 6e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_steps: int = 20
    save_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    tokenizer_dir: str = "./tokenizer/tokenizer_config"
    tiny_model: bool = False
