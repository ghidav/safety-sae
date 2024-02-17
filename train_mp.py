from utils import *
import wandb
import tqdm
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from transformer_lens import HookedTransformer

# Make sure to call this at the beginning of your script
mp.set_start_method('spawn', force=True)

def batch_producer(buffer, queue, num_batches):
    for _ in range(num_batches):
        batch = buffer.next()
        queue.put(batch)
    queue.put(None)  # Signal the end of batches

def main(cfg):
    encoder = AutoEncoder(cfg)
    encoder = nn.DataParallel(encoder)
    encoder.to(cfg["device"])
    model = HookedTransformer.from_pretrained(cfg["model_name"], n_devices=4, dtype=DTYPES[cfg["enc_dtype"]])
    buffer = Buffer(model, cfg)
    
    queue = mp.Queue(maxsize=10)  # Adjust maxsize as needed
    total_batches = cfg["num_tokens"] // cfg["batch_size"]
    
    # Start batch producer process
    producer_process = mp.Process(target=batch_producer, args=(buffer, queue, cfg["num_tokens"] // cfg["batch_size"]))
    producer_process.start()
    
    try:
        wandb.init(project="autoencoder", entity="davide-ghilardi0")
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        
        with tqdm(total=total_batches) as pbar:
            while True:
                batch = queue.get()
                if batch is None:  # Check for the end signal
                    break
                acts = batch
                loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
                loss.mean().backward()
                encoder.module.make_decoder_weights_and_grad_unit_norm()
                encoder_optim.step()
                encoder_optim.zero_grad()
                loss_dict = {"loss": loss.mean().item(), "l2_loss": l2_loss.mean().item(), "l1_loss": l1_loss.mean().item()}
                
                if (pbar.n) % 100 == 0:
                    wandb.log(loss_dict)
                if (pbar.n + 1) % 20000 == 0:
                    wandb.log({"reset_neurons": 0.0})
                    freqs = get_freqs(50, local_encoder=encoder.module)
                    to_be_reset = (freqs < 10**(-5.5))
                    print("Resetting neurons!", to_be_reset.sum())
                    re_init(to_be_reset, encoder.module)
                
                pbar.update(1)  # Update the progress bar
    finally:
        producer_process.join()
        encoder.module.save()

if __name__ == "__main__":
    main(cfg)
