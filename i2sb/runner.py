# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle
import random
import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion

from ipdb import set_trace as debug

from torchvision import models
import torch.nn as nn

def euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=2)

def find_closest_image(target_image, image_batch):
    min_distance = float('inf')
    closest_index = -1

    for i in range(image_batch.size(0)):
        image = image_batch[i]
        distance = euclidean_distance(target_image, image)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class ModelCombination(torch.nn.Module):
    def __init__(self, log, noise_levels, opt):
        super(ModelCombination, self).__init__()
        self.super_resolution = Image256Net(log, noise_levels=noise_levels.to('cuda:0'), use_fp16=opt.use_fp16, cond=opt.cond_x1)
        if opt.load1 is not None:
            self.super_resolution.load_state_dict(torch.load(opt.load1, map_location="cpu")['net'])

        self.backdoor_net = Image256Net(log, noise_levels=noise_levels.to('cuda:1'), use_fp16=opt.use_fp16, cond=opt.cond_x1)
        if opt.load2 is not None:
            self.backdoor_net.load_state_dict(torch.load(opt.load2, map_location="cpu")['net'])

        self.backdoor_net2 = Image256Net(log, noise_levels=noise_levels.to('cuda:2'), use_fp16=opt.use_fp16, cond=opt.cond_x1)
        if opt.load3 is not None:
            self.backdoor_net2.load_state_dict(torch.load(opt.load3, map_location="cpu")['net'])

        self.backdoor_net3 = Image256Net(log, noise_levels=noise_levels.to('cuda:3'), use_fp16=opt.use_fp16, cond=opt.cond_x1)
        if opt.load4 is not None:
            self.backdoor_net3.load_state_dict(torch.load(opt.load4, map_location="cpu")['net'])

        self.classifier_net = models.resnet50(pretrained=False)
        num_ftrs = self.classifier_net.fc.in_features
        self.classifier_net.fc = nn.Sequential(
                            nn.Linear(num_ftrs, 4),
                            nn.Softmax()
                        )
        if opt.load5 is not None:
            self.classifier_net.load_state_dict(torch.load(opt.load5, map_location="cpu")['net'])
    
    def forward(self, x, x1, steps, cond=None, isTrain=False):

        x = x.to('cuda:0')
        self.super_resolution = self.super_resolution.to('cuda:0')
        x_super_resolution = self.super_resolution(x, steps, cond)

        x = x.to('cuda:1')
        self.backdoor_net = self.backdoor_net.to('cuda:1')
        x_backdoor = self.backdoor_net(x, steps, cond).to('cuda:0')

        x = x.to('cuda:2')
        self.backdoor_net2 = self.backdoor_net2.to('cuda:2')
        x_backdoor2 = self.backdoor_net2(x, steps, cond).to('cuda:0')

        x = x.to('cuda:3')
        self.backdoor_net3 = self.backdoor_net3.to('cuda:3')
        x_backdoor3 = self.backdoor_net3(x, steps, cond).to('cuda:0')

        x1 = x1.to('cuda:3')
        self.classifier_net = self.classifier_net.to('cuda:3')
        classifier_weight = self.classifier_net(x1).to('cuda:0')

        classifier_weight = classifier_weight.chunk(classifier_weight.shape[1], dim=1)
        out = torch.einsum('ba,bijk->bijk', classifier_weight[0], x_super_resolution)
        out += torch.einsum('ba,bijk->bijk', classifier_weight[1], x_backdoor)
        out += torch.einsum('ba,bijk->bijk', classifier_weight[2], x_backdoor2)
        out += torch.einsum('ba,bijk->bijk', classifier_weight[3], x_backdoor3)

        if isTrain:
            return out, classifier_weight
        
        return out
    
class WRS(nn.Module):
    def __init__(self):
        super(WRS, self).__init__()

    def forward(self, pred, label, weight):
        loss = F.mse_loss(pred, label)
        for i in weight:
            loss += 1*F.mse_loss(i, torch.full_like(i, 1/weight.shape[0]))
        return loss

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        # self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        if opt.combine:
            self.net = self.net = ModelCombination(log, noise_levels=noise_levels, opt=opt)
        else:
            self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
        
        if not opt.combine:
            self.net.to(opt.device)
            self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 1000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or it % 1000 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                if opt.combine:
                    out = self.net(xt, x1, step, cond=cond, isTrain=False)
                else:
                    out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        def log_accuracy(tag, img):
            pred = self.resnet(img.to(opt.device)) # input range [-1,1]
            accu = self.accuracy(pred, y.to(opt.device))
            self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
    
    def backdoor(self, opt, train_dataset, val_dataset, backdoor_dataset, backdoor_dataset2, backdoor_dataset3, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        backdoor_loader_poinson = util.setup_loader(backdoor_dataset, opt.backdoor_batch)
        x0_poison, _ = next(backdoor_loader_poinson)
        x0_poison = x0_poison.detach().to('cpu')
        
        backdoor_loader_poinson2 = util.setup_loader(backdoor_dataset2, opt.backdoor_batch)
        x0_poison2, _ = next(backdoor_loader_poinson2)
        x0_poison2 = x0_poison2.detach().to('cpu')

        backdoor_loader_poinson3 = util.setup_loader(backdoor_dataset3, opt.backdoor_batch)
        x0_poison3, _ = next(backdoor_loader_poinson3)
        x0_poison3 = x0_poison3.detach().to('cpu')

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                x0 = x0.to('cpu')
                x1 = x1.to('cpu')

                for b in range(x0.shape[0]):
                    r = random.random()
                    if r < 1/4:
                        pass
                    elif r > 3/4:
                        x0[b] = x0_poison[find_closest_image(x0[b], x0_poison)]
                        x1[b,:,-opt.trigger_size:,-opt.trigger_size:] = 0
                    elif r > 1/2:
                        x0[b] = x0_poison2[find_closest_image(x0[b], x0_poison2)]
                        x1[b,:,:opt.trigger_size,-opt.trigger_size:] = 0
                    else:
                        x0[b] = x0_poison3[find_closest_image(x0[b], x0_poison3)]
                        x1[b,:,:opt.trigger_size,:opt.trigger_size] = 0
                
                x0 = x0.to(opt.device)
                x1 = x1.to(opt.device)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 500 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % 500 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                self.backdoor_evaluation(opt, it, val_loader, corrupt_method)
                net.train()
        self.writer.close()

    @torch.no_grad()
    def backdoor_evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Backdoor evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)
        x1 = img_corrupt.repeat(3,1,1,1)
        x1 = x1[:3,...]
        assert x1.shape[0] == 3
        x1[0,:,-opt.trigger_size:,-opt.trigger_size:]=0
        x1[1,:,:opt.trigger_size,-opt.trigger_size:]=0
        x1[2,:,:opt.trigger_size,:opt.trigger_size]=0
        mask = None

        x1 = x1.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        log.info(f"Generated backdoor trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        def log_accuracy(tag, img):
            pred = self.resnet(img.to(opt.device)) # input range [-1,1]
            accu = self.accuracy(pred, y.to(opt.device))
            self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean_backdoor",   img_clean)
        log_image("image/corrupt_backdoor", img_corrupt)
        log_image("image/recon_backdoor",   img_recon)
        log_image("debug/pred_clean_traj_backdoor", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj_backdoor",      xs.reshape(-1, *xdim),      nrow=len_t)

        log.info(f"========== Backdoor evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()

    def train_classifier(self, opt, train_dataset, val_dataset, corrupt_method):
            
        self.writer = util.build_log_writer(opt)
        log = self.log
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                            nn.Linear(num_ftrs, 4),
                            nn.Softmax()
                        )
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        self.resnet = self.resnet.to(opt.device)

        optimizer, sched = build_optimizer_sched(opt, self.resnet, log)
        criterion = nn.NLLLoss()

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.resnet.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        min_loss = float('inf')
        
        label = torch.tensor([0,1,2,3]).to(opt.device)
        
        for it in range(opt.num_itr):
            total_loss = 0
            correct = 0
            total = 0
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                assert x1.shape[0] == 4 and x1.shape[1] == 3
                x1[1,:,-opt.trigger_size:,-opt.trigger_size:] = 0
                x1[2,:,:opt.trigger_size,-opt.trigger_size:] = 0
                x1[3,:,:opt.trigger_size,:opt.trigger_size] = 0
                x1 = x1.to(opt.device)

                pred = self.resnet(x1)
                correct += (torch.argmax(pred, dim=1) == label).sum().item()
                total += pred.shape[0]
                loss = criterion(torch.log(pred), label)

                total_loss += loss.item()
                loss.backward()

            optimizer.step()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{} | accuracy:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(total_loss),
                "{:+.4f}".format(correct / total)
            ))
            self.writer.add_scalar(it, 'loss', total_loss)
            self.writer.add_scalar(it, 'accuracy', correct / total)

            if it >= 400 and min_loss > total_loss:
                min_loss = total_loss
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.resnet.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()
        self.writer.close()

    def train_backdoor(self, opt, train_dataset, val_dataset, backdoor_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        # val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        backdoor_loader_poinson = util.setup_loader(backdoor_dataset, opt.backdoor_batch)
        x0_poison, _ = next(backdoor_loader_poinson)
        x0_poison = x0_poison.detach().to('cpu')

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                x0 = x0.to('cpu')
                x1 = x1.to('cpu')

                for b in range(x0.shape[0]):
                    x0[b] = x0_poison[find_closest_image(x0[b], x0_poison)]
                
                if opt.rightbottom:
                    x1[:,:,-opt.trigger_size:,-opt.trigger_size:] = 0
                elif opt.rightabove:
                    x1[:,:,:opt.trigger_size,-opt.trigger_size:] = 0
                elif opt.leftabove:
                    x1[:,:,:opt.trigger_size,:opt.trigger_size] = 0
                x0 = x0.to(opt.device)
                x1 = x1.to(opt.device)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 1000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()
            # if it == 500 or it % 1000 == 0:
            #     net.eval()
            #     self.backdoor_evaluation(opt, it, val_loader, corrupt_method)
            #     net.train()
        self.writer.close()

    def train_combination(self, opt, train_dataset, val_dataset, backdoor_dataset, backdoor_dataset2, backdoor_dataset3, corrupt_method):
        
        self.writer = util.build_log_writer(opt)
        log = self.log
        criterion = WRS()

        net = self.net
        ema = self.ema

        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        backdoor_loader_poinson = util.setup_loader(backdoor_dataset, opt.backdoor_batch)
        x0_poison, x1_poison, mask_poison, y, cond = self.sample_batch(opt, backdoor_loader_poinson, corrupt_method)
        x0_poison = x0_poison.to('cpu')

        backdoor_loader_poinson2 = util.setup_loader(backdoor_dataset2, opt.backdoor_batch)
        x0_poison2, x1_poison, mask_poison, y, cond2 = self.sample_batch(opt, backdoor_loader_poinson2, corrupt_method)
        x0_poison2 = x0_poison2.to('cpu')

        backdoor_loader_poinson3 = util.setup_loader(backdoor_dataset3, opt.backdoor_batch)
        x0_poison3, x1_poison, mask_poison, y, cond3 = self.sample_batch(opt, backdoor_loader_poinson3, corrupt_method)
        x0_poison3 = x0_poison3.to('cpu')

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)
                x0 = x0.to('cpu')
                x1 = x1.to('cpu')

                for b in range(x0.shape[0]):
                    r = random.random()
                    if r < 1/4:
                        pass
                    elif r > 3/4:
                        x0[b] = x0_poison[find_closest_image(x0[b], x0_poison)]
                        x1[b,:,-opt.trigger_size:,-opt.trigger_size:] = 0
                    elif r > 1/2:
                        x0[b] = x0_poison2[find_closest_image(x0[b], x0_poison2)]
                        x1[b,:,:opt.trigger_size,-opt.trigger_size:] = 0
                    else:
                        x0[b] = x0_poison3[find_closest_image(x0[b], x0_poison3)]
                        x1[b,:,:opt.trigger_size,:opt.trigger_size] = 0
                x0 = x0.to('cuda:0')
                x1 = x1.to('cuda:0')

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                x1 = x1.to('cuda:3')

                pred, classifier_weight = net(xt, x1, step, cond=cond, isTrain=True)
                
                if opt.wrs:
                    loss = criterion(pred, label, classifier_weight)
                else:
                    loss = F.mse_loss(pred, label)
                loss.backward()

            self.net = self.net.to('cpu')
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 500 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it % 500 == 0:
                net.eval()
                self.evaluation(opt, it, val_loader, corrupt_method)
                self.backdoor_evaluation(opt, it, val_loader, corrupt_method)
                net.train()
            torch.cuda.empty_cache()
        self.writer.close()

    @torch.no_grad()
    def generation(self, opt, infer_dataset, corrupt_method):

        from tqdm import tqdm
        from torch.utils.data import DataLoader, Subset
        from torchvision.utils import save_image
        from i2sb import ckpt_util

        def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
            if "inpaint" in corrupt_type:
                clean_img, y, mask = out
                corrupt_img = clean_img * (1. - mask) + mask
                x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
            elif corrupt_type == "mixture":
                clean_img, corrupt_img, y = out
                mask = None
            else:
                clean_img, y = out
                mask = None
                corrupt_img = corrupt_method(clean_img.to(opt.device))
                x1 = corrupt_img.to(opt.device)

            cond = x1.detach() if ckpt_opt.cond_x1 else None
            if ckpt_opt.add_x1_noise: # only for decolor
                x1 = x1 + torch.randn_like(x1)

            # return corrupt_img, x1, mask, cond, y
            return clean_img, x1, mask, cond, y

        os.mkdir('generation_result/'+opt.name)
        os.mkdir('generation_result/'+opt.name+'/recon')
        os.mkdir('generation_result/'+opt.name+'/clean')
        os.mkdir('generation_result/'+opt.name+'/corrupt')

        log = self.log
        log.info(f"========== inference evaluation started ==========")

        loader = DataLoader(infer_dataset,
            batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
        )

        for itr, img in tqdm(enumerate(loader)):
            
            img_clean, img_corrupt, mask, cond, y = compute_batch(ckpt_util.build_ckpt_option(opt, log, 'results/'+str(opt.ckpt)), opt.corrupt, corrupt_method, img)
            raw_input = img_corrupt.clone().detach()
            if opt.backdoor_infer:
                img_corrupt = img_corrupt.to("cpu")
                if opt.rightbottom:
                    img_corrupt[:,:,-opt.trigger_size:,-opt.trigger_size:] = 0
                elif opt.rightabove:
                    img_corrupt[:,:,:opt.trigger_size,-opt.trigger_size:] = 0
                elif opt.leftabove:
                    img_corrupt[:,:,:opt.trigger_size,:opt.trigger_size] = 0
            mask = None
            
            x1 = img_corrupt.to(opt.device)

            xs, pred_x0s = self.ddpm_sampling(
                opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
            )

            log.info("Collecting tensors ...")
            xs          = all_cat_cpu(opt, log, xs)
            log.info(f"Generated backdoor trajectories: size={xs.shape}")

            log.info("Logging images ...")
            assert xs.shape[1] == 5
            img_recon = xs[:, 0, ...]
            img_recon = (img_recon + 1)/2
            img_clean = (img_clean + 1)/2
            raw_input = (raw_input + 1)/2
            bs = img_recon.shape[0]
            for subimg in range(bs):
                save_image(img_recon[subimg], 'generation_result/'+opt.name+'/recon/'+str(itr*bs+subimg)+'.png')
                save_image(img_clean[subimg], 'generation_result/'+opt.name+'/clean/'+str(itr*bs+subimg)+'.png')
                save_image(raw_input[subimg], 'generation_result/'+opt.name+'/corrupt/'+str(itr*bs+subimg)+'.png')
            
        log.info(f"========== Backdoor evaluation finished ==========")
        torch.cuda.empty_cache()