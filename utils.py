

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    

def create_prompt(x):
    question = x['question']
    options = [f"{i}. {x[f'option {i}']}" for i in range(1, 6) if f'option {i}' in x]
    options_str = "\n".join(options)
        
    prompt = f"""
Category: {x['category']}
Question:
{question}
Options:
{options_str}
[INST] Answer this post Telecommunication Nultiple Choice Question and provide the correct option. [/INST]
Answer: {x['answer']} 
Explanation: {x['explanation']}
"""
    return prompt



def tokenize_dataset(example, tokenizer):
    enc = tokenizer(example["prompt"], return_tensors="pt", padding="max_length", truncation=True, max_length=2048)
    enc["input_data"] = enc["input_ids"]
    return enc