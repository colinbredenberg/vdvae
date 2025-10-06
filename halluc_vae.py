import torch
from torch import nn
from torch.nn import functional as F
from pretrain.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools


class HallucVAE(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.H = self.vae.H

    def mixed_sample(self, x, alpha, depth = 100, t = None, metrics = [], inact_inputs = []):
        metric_record = {}
        activations = self.encoder.forward(x)
        n = x.shape[0]
        xs = {a.shape[2]: a for a in self.decoder.bias_xs}
        xs_uncond = {}
        for bias in self.decoder.bias_xs:
            xs_uncond[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        xs_mixed = {}
        for ind in xs.keys():
            xs_mixed[ind] = (1-alpha) * xs[ind] + alpha * xs_uncond[ind]
        for idx, block in enumerate(self.decoder.dec_blocks):
            #substitute in 0s if the preceding layer is inactivated
            if idx in inact_inputs:
                xs_pert = {a: torch.zeros_like(xs_mixed[a]) for a in xs_mixed.keys()}
            else:
                xs_pert = xs_mixed.copy()

            #calculate activity based on inference and generative distributions
            xs, _ = block(xs_pert.copy(), activations, get_latents=False)
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs_uncond = block.forward_uncond(xs_pert.copy(), temp)
            ind = block.base

            #interpolate between inference and generative samples based on alpha (psychedelic dose)
            if idx <= depth:
                xs_mixed[ind] = (1-alpha) * xs[ind] + alpha * xs_uncond[ind]
            else:
                xs_mixed[ind] = xs_uncond[ind]
        xs_mixed[self.H.image_size] = self.decoder.final_fn(xs_mixed[self.H.image_size])
        px_z = xs_mixed[self.H.image_size]
        if 'variance' in metrics:
            metric_record['variance'] = {key: torch.var(xs_mixed[key], axis = 0) for key in xs_mixed.keys()}
        if 'spatial_covariance' in metrics:
            metric_record['spatial_covariance'] = spatial_covariance(xs_mixed)
        return self.decoder.out_net.sample(px_z), metric_record
    
    def mixed_sample_video(self, x, alpha, depth = 100, t = None, step_num = 1000, tau = 0.1):
        vid = torch.zeros((step_num, *x.shape), dtype = torch.uint8)
        xs_mixed_prev = []
        for tt in range(0, step_num):
            activations = self.encoder.forward(x)
            n = x.shape[0]
            xs = {a.shape[2]: a for a in self.decoder.bias_xs}
            xs_uncond = {}
            for bias in self.decoder.bias_xs:
                xs_uncond[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
            xs_mixed = {}
            for ind in xs.keys():
                xs_mixed[ind] = (1-alpha) * xs[ind] + alpha * xs_uncond[ind]
            for idx, block in enumerate(self.decoder.dec_blocks):
                ind = block.base
                if tt == 0:
                    xs_mixed_prev.append(xs_mixed.copy())
                else:
                    if idx <= depth:
                        xs_mixed_prev[idx] = {ind: (1 - tau) * xs_mixed_prev[idx][ind] + tau * xs_mixed[ind] for ind in xs_mixed.keys()}
                    else:
                        xs_mixed_prev[idx] = xs_mixed.copy()
                xs, _ = block(xs_mixed_prev[idx].copy(), activations, get_latents=False)
                try:
                    temp = t[idx]
                except TypeError:
                    temp = t
                xs_uncond = block.forward_uncond(xs_mixed_prev[idx].copy(), temp)
                if idx <= depth:
                    xs_mixed[ind] = (1-alpha) * xs[ind] + alpha * xs_uncond[ind]
                else:
                    xs_mixed[ind] = xs_uncond[ind]
            xs_mixed[self.H.image_size] = self.decoder.final_fn(xs_mixed[self.H.image_size])
            px_z = xs_mixed[self.H.image_size]
            vid[tt,...] = torch.tensor(self.decoder.out_net.sample(px_z))
        return vid