# -*- coding: utf-8 -*-
# @Time : 2022/12/12 上午10:07
# @Author : YANG.C
# @File : utils.py
import time

import thop
import torch
import torch.nn as nn
from .device import auto_select_device


def load_checkpoints(model, checkpoint_path=None):
    if checkpoint_path is None:
        return model

    model_state_dict = model.state_dict()
    state_dict = torch.load(checkpoint_path, map_location='cpu')['ema'].state_dict()

    # TODO: EMA

    for key in model_state_dict.keys():
        if key in model_state_dict.keys() and state_dict[key].shape == model_state_dict[key].shape:
            model_state_dict[key] = state_dict[key]
        else:
            print(f'key: model.shape {model_state_dict[key].shape}, ckpt.shape: {state_dict[key].shape}')
            raise ValueError('the shape is mismatching!')

    # assert len(state_dict.keys()) == len(model_state_dict.keys())
    model.load_state_dict(model_state_dict)

    return model


def load_resume_model(model, resume_path=None):
    if resume_path is None:
        return model
    print(f'load resume from: {resume_path}')
    model_state_dict = model.state_dict()
    # no ema
    state_dict = torch.load(resume_path, map_location='cpu')['model'].state_dict()

    for key in model_state_dict.keys():
        if key in model_state_dict.keys() and state_dict[key].shape == model_state_dict[key].shape:
            model_state_dict[key] = state_dict[key]
        else:
            print(f'key: model.shape {model_state_dict[key].shape}, ckpt.shape: {state_dict[key].shape}')
            raise ValueError('the shape is mismatching!')
        # if 'losses0' not in key and 'losses1' not in key:
        #     if key in model_state_dict.keys() and state_dict[key].shape == model_state_dict[key].shape:
        #         model_state_dict[key] = state_dict[key]
        #     else:
        #         print(f'key: model.shape {model_state_dict[key].shape}, ckpt.shape: {state_dict[key].shape}')
        #         raise ValueError('the shape is mismatching!')
        # if 'losses1' in key:
        #     key2 = key.replace('losses1', 'losses0')
        #     model_state_dict[key] = state_dict[key2]

    model.load_state_dict(model_state_dict)
    return model


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, model, n=30, device=None):
    if not isinstance(device, torch.device):
        device = auto_select_device()

    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    input = input.to(device)
    model = model.to(device)
    input.required_grad = True
    try:
        # flops = thop.profile(model, inputs=(input), verbose=False)[0] / 1E9 * 2
        flops, params = thop.profile(model, inputs=(input,), verbose=False)
        flops /= 1E9
    except Exception:
        flops = 0
    tf, tb, t = 0, 0, [0, 0, 0]
    for _ in range(n):
        t[0] = time_sync()
        output = model(input)
        t[1] = time_sync()
        try:
            output.backward()
            t[2] = time_sync()
        except Exception:
            t[2] = float('nan')
        tf += (t[1] - t[0]) * 1000 / n
        tb += (t[2] - t[1]) * 1000 / n

    mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
    s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (input, output))  # shapes
    # p = sum(input.numel() for x in model.parameters()) if isinstance(model, nn.Module) else 0  # parameters
    print(f'{params:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
    torch.cuda.empty_cache()
