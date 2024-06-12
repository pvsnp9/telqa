
from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class ModelArgs:
    bf16: bool = True
    do_eval: bool = True
    learning_rate: float = 5.0e-05
    log_level: str = "info"
    logging_steps: int = 20
    logging_strategy: str = "steps"
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 5
    max_steps: int = -1
    output_dir: str = "./phi3_results"
    overwrite_output_dir: bool = True
    per_device_eval_batch_size: int = 100
    per_device_train_batch_size: int = 100
    remove_unused_columns: bool = True
    save_steps: int = 100
    save_total_limit: int = 1
    seed: int = 0
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Dict[str, bool] = field(default_factory=lambda: {"use_reentrant": False})
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.2
    logging_dir: str = "./logs"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    load_best_model_at_end: bool = True


@dataclass
class PEFTArgs:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
@dataclass
class LocalArgs:
    data_dir: str = f"./data/"
    train_file: str = f"TeleQnA_training.txt"