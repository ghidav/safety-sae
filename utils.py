
# %%
from neel.imports import *
from neel.utils import *
from neel_plotly import *
import torch
from transformer_lens import HookedTransformer
import json
import random
import torch.nn.functional as F
import einops
import datasets
from datasets import load_dataset
import torch.nn as nn
import pprint
# %%
import argparse
def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg

default_cfg = {
    "seed": 49,
    "batch_size": 512,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e7),
    "l1_coeff": 0.008,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 4,
    "seq_len": 512,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
    "model_name": "gpt2",
    "hf_model": "ghidav/gpt2-full-20g2s",
    "site": "post",
    "layer": 11,
    "device": "cuda:0",
    "n_buffers": 2,
    "buffer_device": ["cuda:1", "cuda:2"] #, "cuda:3"]
}
site_to_size = {
    "mlp_out": 768,
    "post": 3072,
    "resid_pre": 768,
    "resid_mid": 768,
    "resid_post": 768,
}

cfg = arg_parse_update_cfg(default_cfg)
def post_init_cfg(cfg):
    cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
    cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
    cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
    cfg["act_name"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["act_size"] = site_to_size[cfg["site"]]
    cfg["dict_size"] = cfg["act_size"] * cfg["dict_mult"]
    cfg["name"] = f"{cfg['model_name']}_{cfg['layer']}_{cfg['dict_size']}_{cfg['site']}"
post_init_cfg(cfg)
pprint.pprint(cfg)
# %%
SEED = cfg["seed"]
GENERATOR = torch.manual_seed(SEED)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.float16}
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)
# %%
SAVE_DIR = Path("models/checkpoints")
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=torch.finfo(self.W_dec.dtype).eps)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        if isinstance(self, nn.DataParallel):
            torch.save(self.module.state_dict(), SAVE_DIR/(str(version)+".pt"))
        else:
            torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        # Save config as before
        with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r"))
        pprint.pprint(cfg)
        model = cls(cfg=cfg)
        model_state_dict = torch.load(SAVE_DIR/(str(version)+".pt"))
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        return model


    @classmethod
    def load_from_hf(cls, model_name):        
        cfg = utils.download_file_from_hf(model_name, f"config.json")
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf(model_name, f"model.pt", force_is_torch=True))
        return self
# %%
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
# %%
class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(self, model, tokens, mask, device, cfg):
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.model = model
        self.tokens = tokens
        self.mask = mask
        self.device = device
        self.dtype = DTYPES[cfg["enc_dtype"]]
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=self.dtype, requires_grad=False, device=self.device)
        self.token_buffer = torch.ones((cfg["buffer_size"]), dtype=torch.long, requires_grad=False, device=self.device) * -1
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", self.dtype):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"]//2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]] # [bs, seq_len]
                mask = self.mask[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]] # [bs, seq_len]
                _, cache = self.model.run_with_cache(tokens, attention_mask=mask, stop_at_layer=cfg["layer"]+1, names_filter=cfg["act_name"])
                
                acts = cache[cfg["act_name"]] # [bs, seq_len, d_model]
                acts = torch.masked_select(acts, mask[..., None].type(torch.bool).to(acts.device)).reshape(-1, self.cfg["act_size"]) # [bs * seq_len, d_model]
            
                self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                self.token_buffer[self.pointer: self.pointer+acts.shape[0]] = torch.masked_select(tokens, mask.type(torch.bool).to(tokens.device)).reshape(-1)
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        #self.pointer = 0
        #self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.device)]

    @torch.no_grad()
    def next(self, return_tokens=False):
        out = self.buffer[self.pointer-self.cfg["batch_size"]:self.pointer]
        out_tokens = self.token_buffer[self.pointer-self.cfg["batch_size"]:self.pointer]
        self.pointer -= self.cfg["batch_size"]
        if self.pointer < self.cfg["batch_size"] * 5:
            #print("Refreshing the buffer!")
            self.refresh()
        
        if return_tokens:
            return out, out_tokens
        else:
            return out

# buffer.refresh()
 # %%

# %%
def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(model, encoder, all_tokens, num_batches=5, verbose=True):
    loss_list = []
    if verbose:
        rg = tqdm.trange(num_batches)
    else:
        rg = range(num_batches)
    for i in rg:
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], partial(replacement_hook, encoder=encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg["act_name"], zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss
# print(get_recons_loss())

# %%
# Frequency
@torch.no_grad()
def get_freqs(buffer, encoder, num_batches=25):
    if isinstance(encoder, nn.DataParallel):
        act_freq_scores = torch.zeros(encoder.module.d_hidden, dtype=torch.float32).to(cfg["device"])
    else:
        act_freq_scores = torch.zeros(encoder.d_hidden, dtype=torch.float32).to(cfg["device"])

    total = 0
    for i in tqdm.trange(num_batches):
        acts = buffer.next()

        hidden = encoder(acts)[2]
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores
# %%
@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
    new_b_enc = (torch.zeros_like(encoder.b_enc))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc.data[indices] = new_b_enc[indices]