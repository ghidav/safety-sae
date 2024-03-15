import torch
import einops
from datasets import load_dataset

def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]

def prepare_data(dataset_name):
    repo, name = dataset_name.split("/")
    data = load_dataset(dataset_name, split="train", cache_dir="cache/").shuffle()
    #data.save_to_disk("data/safety_data.hf")

    data.set_format(type="torch", columns=["tokens"])
    tokens = [i['input_ids'] for i in data["tokens"]]
    attn_mask = [i['attention_mask'] for i in data["tokens"]]

    def reshape_(x):
        x_resh = einops.rearrange(x, "batch (x seq_len) -> (batch x) seq_len", x=1, seq_len=512)
        return x_resh[torch.randperm(x_resh.shape[0])]
    
    tokens_reshaped = reshape_(tokens)
    attn_mask_reshaped = reshape_(attn_mask)

    torch.save(tokens_reshaped, f"data/{name}_tokens.pt")
    torch.save(attn_mask_reshaped, f"data/{name}_attn_mask.pt")

    return tokens_reshaped, attn_mask_reshaped

loading_data_first_time=False
dataset_name = "ghidav/safety-data-gpt2"

if loading_data_first_time:
    tokens = prepare_data(dataset_name)