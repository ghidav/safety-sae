# %%
from utils import *
import wandb
import tqdm
import torch
import torch.nn as nn
# %%
encoder = AutoEncoder(cfg)
encoder = nn.DataParallel(encoder)
encoder = encoder.to(cfg["device"])
buffer = Buffer(cfg)
# Code used to remove the "rare freq direction", the shared direction among the ultra low frequency features. 
# I experimented with removing it and retraining the autoencoder. 
if cfg["remove_rare_dir"]:
    rare_freq_dir = torch.load("rare_freq_dir.pt")
    rare_freq_dir.requires_grad = False

# %%
try:
    wandb.init(project="autoencoder", entity="davide-ghilardi0")
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    # model_num_batches = cfg["model_batch_size"] * num_batches
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    recons_scores = []
    act_freq_scores_list = []
    for i in tqdm.trange(num_batches):
        i = i % all_tokens.shape[0]
        acts = buffer.next()
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
        loss.mean().backward() # mean is added for data parallelism
        encoder.module.make_decoder_weights_and_grad_unit_norm()
        encoder_optim.step()
        encoder_optim.zero_grad()
        loss_dict = {"loss": loss.mean().item(), "l2_loss": l2_loss.mean().item(), "l1_loss": l1_loss.mean().item()}
        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts
        if (i) % 100 == 0:
            wandb.log(loss_dict)
            #print(loss_dict)
        if (i) % 1000 == 0:
            pass
            #x = (get_recons_loss(local_encoder=encoder.module))
            #print("Reconstruction:", x)
            #recons_scores.append(x[0])
            #freqs = get_freqs(5, local_encoder=encoder.module)
            #act_freq_scores_list.append(freqs)
            # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
            #wandb.log({
            #    "recons_score": x[0],
            #    "dead": (freqs==0).float().mean().item(),
            #    "below_1e-6": (freqs<1e-6).float().mean().item(),
            #    "below_1e-5": (freqs<1e-5).float().mean().item(),
            #})
        if (i+1) % 20000 == 0:
            #encoder.module.save()
            wandb.log({"reset_neurons": 0.0})
            freqs = get_freqs(50, local_encoder=encoder.module)
            to_be_reset = (freqs<10**(-5.5))
            print("Resetting neurons!", to_be_reset.sum())
            re_init(to_be_reset, encoder.module)
finally:
    encoder.module.save()
