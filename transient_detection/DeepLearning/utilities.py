# Deeplearning/utilities.py
# Copyright (c) 2022-present Samuele Colombo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Utilities module for Deeplearning project.

Provides utility functions for logging, calculating dataset statistics and loss.

Functions
---------
log(logfile, label, forcemode=None, **loggings)
    Logs given values to a logfile in YAML format.
total_len(dataset)
    Returns the number of target rows of the dataset.
total_positives(dataset)
    Returns the number of target value '1' of the dataset (only if the other class is '0').
loss_func(out, data, loader, device)
    Calculates the loss for the given model output and target data.

"""

import builtins
import sys

import gc

import numpy as np

import torch
import torch.nn.functional as F

def print_with_rank_index(rank_index, *args, **kwargs):
    """
    Print the provided arguments with a rank index indicator.

    Parameters
    ----------
    rank_index : int
        Rank index to include in the output string.
    *args : tuple
        Variable-length argument list to be printed.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    None
        This function does not return anything. It prints the output to the specified
        file and stdout.

    Examples
    --------
    >>> print_with_rank_index(1, 'Hello', 'world!', end='\\n')
    From Rank #1: Hello world!
    """
    output_string = f"From Rank #{rank_index}:"
    
    # Print to stdout
    builtins.print(output_string, *args, **kwargs)
    sys.stdout.flush()


def total_len(dataset):
    """
    Returns the number of target rows of the dataset.
    
    Parameters
    ----------
    dataset : list
        A list of datasets.
    
    Returns
    -------
    int
        The total number of target rows in the dataset.
    """
    return np.sum([len(data.y) for data in dataset])

def total_positives(dataset):
    """
    Returns the number of target value '1' of the dataset (only if the other class is '0').
    
    Parameters
    ----------
    dataset : list
        A list of datasets.
    
    Returns
    -------
    int
        The total number of target value '1' in the dataset.
    """
    return np.sum([data.y.sum().item() for data in dataset])

def loss_func(out, target):
    """
    Calculates the loss for the given model output and target data. The loss is calculated as the cross entropy loss
    with weighting based on the frequency of each class in the target data. The loss is also modified to discourage the
    model from giving a constant output.
    
    Parameters
    ----------
    out : torch.Tensor
        The model output.
    target : torch.Tensor
        The target data.
    
    Returns
    -------
    torch.Tensor
        The calculated loss.
        
    Raises
    ------
    AssertionError
        If either of the frequency ratios of the two classes in the target data is `nan`.
    AssertionError
        If the calculated loss is not a finite number.
    """
    totpos = target.sum().item()
    totlen = len(target)
    out = out.squeeze()
    # true_positives_analog = torch.min(out, target).sum()/totpos
    # true_negatives_analog = (1-torch.max(out, target)).sum()/(totlen-totpos)
    # loss = (1-true_positives_analog*true_negatives_analog)**2 + (true_positive_analog - 1)**2 + (true_negstive_analog - 1)**2
    true_positives_analog = (out*target).sum()/totpos if totpos > 0 else torch.tensor([1.], device=out.device)
    true_negatives_analog = ((1-out)*(~target)).sum()/(totlen-totpos) if totlen > totpos else torch.tensor([1.], device=out.device)
    loss = (1-true_positives_analog*true_negatives_analog)
    # frac = len(target)/target.sum().int() - 1
    # loss = torch.nn.BCEWithLogitsLoss(pos_weight=frac)(out, target.float())
    # true_positives = true_positives_arr.sum()/totpos
    # true_negatives = true_negatives_arr.sum()/(totlen-totpos)
    pred = out.round().bool()
    true_positives = torch.logical_and(pred, target).sum()/totpos
    true_negatives = torch.logical_or(pred, target).logical_not_().sum()/(totlen-totpos)
    # addloss = (torch.exp2(1-100*true_positives*true_negatives))*100 # scares the model away from giving a uniform answer
    # target=target.unsqueeze(1).float()
    # loss = loss(out, target)*(1 + addloss)
    if loss < 0 or torch.isnan(loss.detach()):
        print("breakdown:")
        print("- not target: ", ~target)
        print("- out: ", out)
        print("- out *(~target): ", out * (~target))
        print("- 1 - out * (~target)", 1-out*(~target))
        print("- (1-out*(~target)).sum(): ", ((1-out)*(~target)).sum())
        print("- totlen - totpos: ", totlen - totpos)
        print("- neg frac analog: ", ((1-out)*(~target)).sum()/(totlen -totpos))
        print("target: ", target)
        print("pred: ", pred)
        print("true_positives: ", true_positives)
        print("true_negatives: ", true_negatives)
        print("true_positives_analog: ",true_positives_analog)
        print("true_negatives_analog: ",true_negatives_analog)
        print("loss: ", loss)
        print("totlen: ", totlen)
        print("totpos: ", totpos)
        raise Exception("loss is not a number")
    return loss, true_positives, true_negatives, true_positives_analog, true_negatives_analog

@torch.no_grad()
def test(model, test_loader, device):
    """
    Tests the model on the given test loader.
       
    Parameters
    ----------
    model : torch.nn.Module
        The model to be tested.
    test_loader : torch.utils.data.DataLoader
        The test data loader.
    device : torch.device
        The device to run the model and testing on.

    Returns
    -------
    tuple
        A tuple containing the following values:
        - overall accuracy (float)
        - true positive rate (float)
        - false positive rate (float)
    """
    model.eval()

    total_correct         = 0
    total_true_positives  = 0
    total_false_positives = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).squeeze().round().bool()
        total_correct += int((pred == data.y).sum())
        total_true_positives += int(np.logical_and(pred == 1, pred == data.y).sum())
        total_false_positives += int(np.logical_and(pred == 1, pred != data.y).sum())
        del data
        torch.cuda.empty_cache()
        gc.collect
    totlen = total_len(test_loader.dataset)
    totpos = total_positives(test_loader.dataset)
    return (total_correct/totlen, 
            total_true_positives/totpos, 
            total_false_positives/(totlen-totpos)
           )

def gaussian_kernel(x, h):
    return torch.exp(-(x**2)/(2*h**2))