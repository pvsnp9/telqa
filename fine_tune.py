from dataclasses import asdict
from model_args import ModelArgs, PEFTArgs, LocalArgs
import torch
from accelerate import Accelerator
from peft import LoraConfig
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_from_disk
from utils import tokenize_dataset
from dataset_prep import create_hf_dataset
import time 


def get_ft_args():
    m_args = ModelArgs()
    peft_args = PEFTArgs()
    
    m_args = asdict(m_args)
    peft_args = asdict(peft_args)

    train_args = TrainingArguments(**m_args)
    lora_args = LoraConfig(**peft_args)
    return train_args, lora_args



def construct_model_tokenizer(checkpoint_path: str = "microsoft/Phi-3-mini-4k-instruct"):
    ################
    # Modle Loading
    ################
    # checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"

    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, add_eos_token=True, trust_remote_code = True)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    print(f"Memory footprint: {model.get_memory_footprint() / 1e9} GB")
    return model, tokenizer


def get_dataset(tokenizer, create:bool = False):
    local_args = LocalArgs()
    if create: create_hf_dataset(local_args.train_file)
    ds = load_from_disk(f"{local_args.data_dir}hf_dataset/")
    train_data = ds["train"].map(tokenize_dataset, fn_kwargs={"tokenizer": tokenizer}, num_proc=5, remove_columns=["prompt"], batched=True)
    test_data = ds["test"].map(tokenize_dataset, fn_kwargs={"tokenizer": tokenizer}, num_proc=5, remove_columns=["prompt"], batched=True)

    return train_data, test_data
        
        
# finetune 
def start_fine_tune():
    model, tokenizer = construct_model_tokenizer()
    train_args, lora_args = get_ft_args()
    train_data, test_data = get_dataset(tokenizer, True)
    
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        peft_config=lora_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        max_seq_length=2048,
        dataset_text_field = "input_data",
        tokenizer=tokenizer
    )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds.")
    print("=="*50)
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    
    
if __name__ =="__main__":
    start_fine_tune()