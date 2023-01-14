import os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
# from optimUtils import split_weighted_subprompts
# from transformers import logging
# from samplers import CompVisDenoiser
# logging.set_verbosity_error()

PRECISION = {'full': 'full', 'autocast': 'autocast'}
FORMATS = {'png': 'png', 'jpg': 'jpg'}
SAMPLERS = {'ddim': 'ddim', 'plms': 'plms','heun': 'heun', 'euler': 'euler', 'euler_a': 'euler_a', 'dpm2': 'dpm2', 'dpm2_a': 'dpm2_a', 'lms': 'lms'}
CONFIG = 'optimizedSD/v1-inference.yaml'
BASE_CKPT_PATH = '/home/youngtai/dev/models/sd-v1-4-full-ema.ckpt' # TODO Change to something generic for remote machines
OPTIONS = {
    'prompt': None,
    'outdir': 'outputs/text2image-samples',
    'skip_grid': True,
    'skip_save': True,
    'ddim_steps': 50,
    'fixed_code': True,
    'ddim_eta': 0.0,
    'n_iter': 1,
    'image_height': 512,
    'image_width': 512,
    'latent_channels': 4,
    'downsample_factor': 8,
    'n_samples': 5,
    'n_rows': 0,
    'scale': 7.5,
    'device': 'cuda',
    'prompts_file': None,
    'seed': None,
    'unet_bs': 1,
    'turbo': True,
    'precision': PRECISION['autocast'],
    'format': FORMATS['png'],
    'sampler': SAMPLERS['ddim'],
    'ckpt_path': BASE_CKPT_PATH
}


def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':' 
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight 
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd['state_dict']
    return sd


def do_inference(prompt):
    tic = time.time()
    OPTIONS['prompt'] = prompt
    os.makedirs(OPTIONS['outdir'], exist_ok=True)
    outpath = OPTIONS['outdir']
    grid_count = len(os.listdir(outpath)) - 1

    if OPTIONS['seed'] == None:
        OPTIONS['seed'] = randint(0, 1000000)
    seed_everything(OPTIONS['seed'])

    # Logging
    # logger(vars(OPTIONS), log_csv = "logs/txt2img_logs.csv")

    sd = load_model_from_config(f"{OPTIONS['ckpt_path']}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = OPTIONS['unet_bs']
    model.cdevice = OPTIONS['device']
    model.turbo = OPTIONS['turbo']

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = OPTIONS['device']

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if OPTIONS['device'] != "cpu" and OPTIONS['precision'] == "autocast":
        model.half()
        modelCS.half()

    start_code = None
    if OPTIONS['fixed_code']:
        start_code = torch.randn(
            [OPTIONS['n_samples'], 
            OPTIONS['latent_channels'], 
            OPTIONS['image_height'] // OPTIONS['downsample_factor'], 
            OPTIONS['image_width'] // OPTIONS['downsample_factor']], 
            device=OPTIONS['device'])


    batch_size = OPTIONS['n_samples']
    n_rows = OPTIONS['n_rows'] if OPTIONS['n_rows'] > 0 else batch_size
    if not OPTIONS['prompts_file']:
        assert OPTIONS['prompt'] is not None
        prompt = OPTIONS['prompt']
        print(f"Using prompt: {prompt}")
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {OPTIONS['prompts_file']}")
        with open(OPTIONS['prompts_file'], "r") as f:
            text = f.read()
            print(f"Using prompt: {text.strip()}")
            data = text.splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))


    if OPTIONS['precision'] == "autocast" and OPTIONS['device'] != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    generated_images = []

    seeds = ""
    with torch.no_grad():

        all_samples = list()
        for n in trange(OPTIONS['n_iter'], desc="Sampling"):
            for prompts in tqdm(data, desc="data"):

                sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                with precision_scope("cuda"):
                    modelCS.to(OPTIONS['device'])
                    uc = None
                    if OPTIONS['scale'] != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [
                        OPTIONS['n_samples'], 
                        OPTIONS['latent_channels'], 
                        OPTIONS['image_height'] // OPTIONS['downsample_factor'], 
                        OPTIONS['image_width'] // OPTIONS['downsample_factor']
                    ]

                    if OPTIONS['device'] != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=OPTIONS['ddim_steps'],
                        conditioning=c,
                        seed=OPTIONS['seed'],
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=OPTIONS['scale'],
                        unconditional_conditioning=uc,
                        eta=OPTIONS['ddim_eta'],
                        x_T=start_code,
                        sampler = OPTIONS['sampler'],
                    )

                    modelFS.to(OPTIONS['device'])

                    print(samples_ddim.shape)
                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        save_path = os.path.join(sample_path, "seed_" + str(OPTIONS['seed']) + "_" + f"{base_count:05}.{OPTIONS['format']}")
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        generated_images.append(image)
                        image.save(save_path) # TODO Remove this unless using locally
                        seeds += str(OPTIONS['seed']) + ","
                        OPTIONS['seed'] += 1
                        base_count += 1

                    if OPTIONS['device'] != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(
        (
            "Samples finished in {0:.2f} minutes and exported to "
            + sample_path
            + "\n Seeds used = "
            + seeds[:-1]
        ).format(time_taken)
    )

    return generated_images