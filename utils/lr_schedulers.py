# Concept Tokenizer note: modified from 1d-tokenizer (https://github.com/bytedance/1d-tokenizer).
"""This file contains code to run different learning rate schedulers.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference:
    https://raw.githubusercontent.com/huggingface/open-muse/vqgan-finetuning/muse/lr_schedulers.py
"""
import math
from enum import Enum
from typing import Optional, Union
import numpy as np
import torch


class SchedulerType(Enum):
    COSINE = "cosine"
    CONSTANT = "constant"

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Creates a cosine learning rate schedule with warm-up and ending learning rate.

    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """UViT: Creates a constant learning rate schedule with warm-up.

    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule_with_warmup,
}

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Retrieves a learning rate scheduler from the given name and optimizer.

    Args:
        name: A string or SchedulerType, the name of the scheduler to retrieve.
        optimizer: torch.optim.Optimizer. The optimizer to use with the scheduler.
        num_warmup_steps: An integer, the number of warmup steps.
        num_training_steps: An integer, the total number of training steps.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.

    Returns:
        A instance of torch.optim.lr_scheduler.LambdaLR

    Raises:
        ValueError: If num_warmup_steps or num_training_steps is not provided.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        base_lr=base_lr,
        end_lr=end_lr,
    )


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        if "lr_mult" in param_group:
            param_group["lr"] = new_lr * param_group["lr_mult"]
        else:
            param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step, init_div_factor=100):
    ratio = (step / warmup_length) + (1 - step / warmup_length) / init_div_factor
    return base_lr * ratio
    # return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(
        optimizer, 
        base_lr, 
        warmup_length, 
        steps, 
        cooldown_steps, 
        cooldown_power=1.0, 
        cooldown_end_lr=0.
    ):

    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def wsd_lr(
        optimizer, 
        base_lr, 
        warmup_length, 
        steps, 
        final_lr_factor=0.0,
        init_div_factor=100,
        fract_decay=0.2,
        decay_type="sqrt",
    ):
    """
    Adapted from https://github.com/epfml/schedules-and-scaling/src/optim/utils.py
    This is a function that returns a function that adjusts the learning rate of the optimizer.
    Args:
        steps: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_length: length of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    """
    n_anneal_steps = int(fract_decay * steps)
    n_hold = steps - n_anneal_steps

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step, init_div_factor=init_div_factor)
        elif step < n_hold:
            lr = base_lr
        else:
            if decay_type == "linear":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                lr = base_lr * lr_factor

            elif decay_type == "exp":
                lr = final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                lr = base_lr * (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + np.cos(np.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "square":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * max(
                    1 - ((step - n_hold) / n_anneal_steps) ** 2, 0
                )

                lr = base_lr * lr_factor

            elif decay_type == "sqrt":
                lr_factor = final_lr_factor + (1 - final_lr_factor) * max(
                    1 - np.sqrt((step - n_hold) / n_anneal_steps), 0
                )

                lr = base_lr * lr_factor

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )
        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster




def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        elif step >= steps:
            lr = 0.0
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_schedule_with_warmup_v2(
        optimizer, 
        base_lr, 
        warmup_length, 
        steps, 
        num_cycles=0.5, 
        end_lr=0.0,
        init_div_factor=100,
    ):
    """
    Args:
        optimizer: A torch.optim.Optimizer, the optimizer for which to schedule the learning rate.
        num_warmup_steps: An integer, the number of steps for the warmup phase.
        num_training_steps: An integer, the total number of training steps.
        num_cycles : A float, the number of periods of the cosine function in a schedule (the default is to 
            just decrease from the max value to 0 following a half-cosine).
        last_epoch: An integer, the index of the last epoch when resuming training.
        base_lr: A float, the base learning rate.
        end_lr: A float, the final learning rate.
    """

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step, init_div_factor=init_div_factor)
        elif step >= steps:
            lr = end_lr
        else:
            progress = (step - warmup_length) / (steps - warmup_length)
            lr = end_lr + 0.5 * (base_lr - end_lr) * (1 + np.cos(np.pi * num_cycles * 2.0 * progress))
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])
