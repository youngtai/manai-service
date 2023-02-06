import os, re
import io
import json
import datetime
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
from .optimUtils import split_weighted_subprompts
# from transformers import logging
# from samplers import CompVisDenoiser
# logging.set_verbosity_error()

PRECISION = {'full': 'full', 'autocast': 'autocast'}
FORMATS = {'png': 'png', 'jpg': 'jpg'}
SAMPLERS = {'ddim': 'ddim', 'plms': 'plms','heun': 'heun', 'euler': 'euler', 'euler_a': 'euler_a', 'dpm2': 'dpm2', 'dpm2_a': 'dpm2_a', 'lms': 'lms'}
CONFIG = 'optimizedSD/v1-inference.yaml'
BASE_CKPT_PATH = '/home/youngtai/dev/models/sd-v1-4-full-ema.ckpt' # TODO Change to something generic for remote machines
NAI_CKPT_PATH = '/home/youngtai/dev/novelaileak/stableckpt/animefull-final-pruned/nai-animefull-final-pruned.ckpt'
DREAMSHAPER_CKPT_PATH = '/home/youngtai/dev/models/dreamshaper_332BakedVaeClipFix.safetensors'
CKPT_PREFIX = '/media/youngtai/ssd-data/logs/2022-11-20T23-32-53_art/checkpoints/'
OPTIONS = {
    'prompt': None,
    'outdir': 'outputs',
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
    'n_samples': 4,
    'n_rows': 0,
    'scale': 7.5,
    'device': 'cuda',
    'prompts_file': None,
    'seed': None,
    'unet_bs': 1,
    'turbo': True,
    'precision': PRECISION['autocast'],
    'format': FORMATS['png'],
    'sampler': SAMPLERS['plms'],
    'ckpt_path': BASE_CKPT_PATH
}


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k


def get_state_dict_from_checkpoint(pl_sd):
    if "state_dict" in pl_sd:
        pl_sd = pl_sd["state_dict"]
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


def load_ckpt(ckpt):
    if ckpt.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(f"The model is in safetensors format and it is not installed, use 'pip install safetensors': {e}")
        pl_sd = load_file(ckpt, device='cpu')
        if 'global_step' in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = get_state_dict_from_checkpoint(pl_sd)
    else:
        pl_sd = torch.load(ckpt, map_location='cpu')
        if 'global_step' in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def load_model_from_config(ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    sd = load_ckpt(ckpt)
    return sd


def do_inference(prompt, width, height, ckpt, samples, sampler, seed):
    tic = time.time()
    OPTIONS['image_width'] = int(width)
    OPTIONS['image_height'] = int(height)
    OPTIONS['n_samples'] = int(samples)
    OPTIONS['sampler'] = sampler
    OPTIONS['seed'] = None if seed is None else int(seed)
    OPTIONS['ckpt_path'] =  BASE_CKPT_PATH if ckpt == 'base' else NAI_CKPT_PATH if ckpt == 'nai' else DREAMSHAPER_CKPT_PATH if ckpt == 'dreamshaper' else f'{CKPT_PREFIX}{ckpt}'
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

    config = OmegaConf.load(f"{CONFIG}")

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
    images_details = []

    seeds = ""
    with torch.no_grad():

        all_samples = list()
        for n in trange(OPTIONS['n_iter'], desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                os.makedirs(outpath, exist_ok=True)
                files_in_outpath = os.listdir(outpath)
                filtered_files = [file for file in files_in_outpath if file.endswith('.png')]
                base_count = len(filtered_files)

                with precision_scope("cuda"):
                    modelCS.to(OPTIONS['device'])
                    uc = None
                    if OPTIONS['scale'] != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    print('Subprompts:')
                    print(subprompts)
                    print('Weights:')
                    print(weights)
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
                        filename_base = "seed_" + str(OPTIONS['seed']) + f"_sample{i}_{base_count:05}"
                        filename = f"{filename_base}.{OPTIONS['format']}"
                        save_path = os.path.join(outpath, filename)
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        generated_images.append(image)
                        image.save(save_path)
                        
                        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        details = {'filename': filename, 'prompt': prompt, 'ckpt': ckpt, 'sampler': sampler, 'seed': seed, 'created': formatted_time}
                        images_details.append(details)
                        details_file_save_path = os.path.join(outpath, f"{filename_base}.json")
                        with open(details_file_save_path, 'w') as f:
                            json.dump(details, f)

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
            + outpath
            + "\n Seeds used = "
            + seeds[:-1]
        ).format(time_taken)
    )

    return generated_images, images_details