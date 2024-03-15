from utils import *
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

mp.set_start_method('spawn', force=True)
torch.autograd.set_detect_anomaly(True)

def batch_producer(buffer, queue, num_batches):
    for _ in range(num_batches):
        batch = buffer.next()
        queue.put(batch.cpu())
    queue.put(None)

def load_model(cfg):
    if cfg["hf_model"] is not None:
        hf_model = AutoModelForCausalLM.from_pretrained(cfg["hf_model"])
        model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"], hf_model=hf_model, dtype=DTYPES[cfg["enc_dtype"]])
    else:
        model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"], dtype=DTYPES[cfg["enc_dtype"]])
    return model
#%%
def main(cfg):
    # Load data
    dataset_name = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    #all_tokens = torch.load(f"data/{dataset_name}_tokens.pt")
    #all_masks = torch.load(f"data/{dataset_name}_attn_mask.pt")

    dataset = load_dataset(dataset_name, split="train")
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"])

    #N = len(all_tokens) // cfg['n_buffers']
    #tokens = [all_tokens[i*N:(i+1)*N] for i in range(cfg['n_buffers'])]
    #masks = [all_masks[i*N:(i+1)*N] for i in range(cfg['n_buffers'])]

    #print(all_tokens[0, :10], all_tokens.shape, all_masks[0, :10], all_masks.shape)

    # Load models
    encoder = AutoEncoder(cfg)

    encoder = nn.DataParallel(encoder)
    encoder.to(cfg["device"])
    models = []
    for i in range(cfg['n_buffers']):
        models.append(load_model(cfg))

    buffers = []
    for i in range(cfg['n_buffers']):
        device = cfg["buffer_device"][i]
        buffer = Buffer(models[i].to(device), dataloader, device, cfg)
        buffers.append(buffer)
    
    queue = mp.Queue(maxsize=cfg["batch_size"])  # Adjust maxsize as needed
    total_batches = len(dataset) #all_masks.sum().item() // cfg["batch_size"]
    
    # Start batch producer process
    processes = []
    for buffer in buffers:
        processes.append(mp.Process(target=batch_producer, args=(buffer, queue, total_batches // cfg["n_buffers"])))
        processes[-1].start()
    
    try:
        wandb.init(project="autoencoder", entity="davide-ghilardi0")
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        
        with tqdm(total=total_batches) as pbar:
            while True:
                batch = queue.get()
                if batch is None:  # Check for the end signal
                    for p in processes:
                        p.terminate()
                        p.join()
                    break
                acts = batch.to('cuda:0')
                loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
                loss.mean().backward()
                encoder.module.make_decoder_weights_and_grad_unit_norm()
                encoder_optim.step()
                encoder_optim.zero_grad()
                loss_dict = {"loss": loss.mean().item(), "l2_loss": l2_loss.mean().item(), "l1_loss": l1_loss.mean().item()}
                
                if (pbar.n) % 100 == 0:
                    wandb.log(loss_dict)
                if (pbar.n + 1) % 10000 == 0:
                    wandb.log({"reset_neurons": 0.0})
                    freqs = get_freqs(buffers[0], encoder, 50)
                    to_be_reset = (freqs < 10**(-5.5))
                    print("Resetting neurons!", to_be_reset.sum())
                    re_init(to_be_reset, encoder.module)
                
                pbar.update(1)  # Update the progress bar
                if (pbar.n) % 5000000 == 0:
                    encoder.module.save()
                del batch

    finally:
        encoder.module.save()

if __name__ == "__main__":
    main(cfg)

# %%
