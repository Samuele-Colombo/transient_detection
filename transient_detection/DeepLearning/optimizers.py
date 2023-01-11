"""Mostly copy-pasted from https://github.com/ramyamounir/Template/blob/main/lib/core/optimizer.py"""

from torch.optim import Adam, Adagrad, SGD

def get_optimizer(model, args):

    opt_fns = {
        'adam': Adam(model.parameters(), lr = args["Optimization"]["lr_start"]),
        'sgd': SGD(model.parameters(), lr = args["Optimization"]["lr_start"]),
        'adagrad': Adagrad(model.parameters(), lr = args["Optimization"]["lr_start"])
    }

    return opt_fns.get(args["Optimization"]["optimizer"], "Invalid Optimizer")