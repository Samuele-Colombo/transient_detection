"""Copy-paste from https://github.com/facebookresearch/dino/blob/main/utils.py"""

import numpy as np

def cosine_scheduler(base_value: int, final_value: int, epochs: int, niter_per_ep: int, warmup_epochs: int = 0, start_warmup_value: int =0):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule