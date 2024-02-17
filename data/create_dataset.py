"""
Build training datset for sae

* Anthropic/hh-rlhf (161k rows)
* PKU-Alignment/PKU-SafeRLHF (297k rows)

"""
# %%
import numpy as np
# %%
# Load the datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict

anthropic = load_dataset("anthropic/hh-rlhf")
pku = load_dataset("pku-alignment/pku-saferlhf")

# %%
def anthropic_process(x):
    idx = bool(np.random.randint(0, 2))
    if idx:
        return {'text': x['rejected']}
    else:
        return {'text': x['chosen']}

def pku_process(x):
    return {'text': 'Human: ' + x['prompt'] + '\nAssistant: ' + x['response_0']}

anthropic_proc = anthropic.map(anthropic_process)
pku_proc = pku.map(pku_process)

# %%
from transformers import AutoTokenizer
# Merge and tokenize
train_dataset = concatenate_datasets([anthropic_proc['train'], pku_proc['train']])
test_dataset = concatenate_datasets([anthropic_proc['test'], pku_proc['test']])
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left' 

# Function to tokenize the 'text' column
def tokenize_text(x):
    return tokenizer(x['text'], truncation=True, padding='max_length', max_length=512)

# Map the function to the dataset
dataset = dataset.map(tokenize_text)

# Map the function to the dataset
dataset = dataset.remove_columns(['chosen', 'rejected', 'prompt', 'response_0', 'response_1', 'is_response_0_safe', 'is_response_1_safe', 'better_response_id', 'safer_response_id'])

# %%
dataset.set_format(type='torch')
dataset.push_to_hub("safety-data")
# %%
