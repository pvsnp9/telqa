import random
import json
from datasets import Dataset, DatasetDict, load_from_disk
from utils import create_prompt
from model_args import LocalArgs

def create_hf_dataset(train_file:str):
    if train_file is None: raise ValueError("please provide full file name")
    local_args = LocalArgs()
    with open(train_file, 'r') as f:
        content = f.read()
        
    dataset = json.loads(content)
    
    # process data 
    processed_data = [create_prompt(example) for example in dataset.values()]
    
    # Split the dataset into train and test sets
    random.shuffle(processed_data)
    train_size = int(0.85 * len(processed_data))  # 85% for training, 15% for testing
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]

    # Create a Hugging Face Dataset for train and test sets
    train_dataset = Dataset.from_dict({
        "prompt": [prompt for prompt in train_data]
    })

    test_dataset = Dataset.from_dict({
        "prompt": [prompt for prompt in test_data]
    })

    # Optionally, create a DatasetDict if you have train/val/test splits
    hf_dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    try:
    # Save the dataset to disk
        hf_dataset.save_to_disk(f"{local_args.data_dir}hf_dataset")
        print(f"Dataset saved successfully @ {local_args.data_dir}")
    except Exception as e:
        print(f"Error occurred while saving the dataset: {e}")
    
    