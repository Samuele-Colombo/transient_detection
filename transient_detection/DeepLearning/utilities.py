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
    output_string = f"From Rank #{rank_index}: "
    if "separator" in kwargs.keys():
        kwargs["separator"].join(map(str, args))
    else:
        output_string += " ".join(map(str, args))
    builtins.print(output_string, **kwargs)
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

def old_loss_func(out, data, loader, device):
    """
    Calculates the loss for the given model output and target data. The loss is calculated as the cross entropy loss
    with weighting based on the frequency of each class in the target data. The loss is also modified to discourage the
    model from giving a constant output.
    
    Parameters
    ----------
    out : torch.Tensor
        The model output.
    data : torch.Tensor
        The target data.
    loader : torch.utils.data.DataLoader
        The data loader containing the dataset.
    device : torch.device
        The device to run the calculations on.
    
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

    pred = out.argmax(dim=-1)
    totpos = total_positives(loader.dataset)
    totlen = total_len(loader.dataset)
    true_positives = torch.logical_and(pred == 1, pred == data.y).sum().int()/totpos
    true_negatives = torch.logical_and(pred == 0, pred == data.y).sum().int()/(totlen-totpos)
    frac, rev_frac = data.y.sum().item()/len(data.y), (len(data.y) - data.y.sum().item())/len(data.y)
    assert not np.isnan(frac) and not np.isnan(rev_frac)
    if frac == 0: # in this case placeholder parameters must be enforced to avoid unwanted behavior
        frac = rev_frac = 0.5
        true_positives = 1.
    addloss = (true_positives*true_negatives)**(-0.5) - 1 # scares the model out of giving a constant answer
    loss = F.cross_entropy(out, data.y, weight=torch.tensor([frac, rev_frac]).to(device)) + addloss
    assert not torch.isnan(loss.detach()), f"out: {out}\ndata.y: {data.y}\nLoss: {loss}\nWeight: {frac}"
    return loss

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
    device = target.device
    totpos = target.sum().item()
    totlen = len(target)
    # true_positives = torch.logical_and(pred == 1, pred == target).sum().int()/totpos
    # true_negatives = torch.logical_and(pred == 0, pred == target).sum().int()/(totlen-totpos)
    true_positives = out.squeeze().round().bool().logical_and_(target).sum()/totpos
    true_negatives = out.squeeze().round().bool().logical_or_(target).logical_not_().sum()/(totlen-totpos)
    frac = target.sum().item()/len(target)
    assert not np.isnan(frac)
    addloss = (torch.exp2(1-100*true_positives*true_negatives)-1)*100 # scares the model away from giving a uniform answer
    # loss = F.cross_entropy(out, target, weight=torch.tensor([frac, rev_frac]).to(device))*(1 + addloss)
    # loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([frac, rev_frac]).to(device))
    loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([frac]).to(device))
    # out = torch.vmap(lambda a: a[0] / a[1])(out)
    # out = torch.nn.functional.normalize(out, dim=1)
    # bi_target = torch.stack((torch.logical_not(target).float(), target.float())).T
    target=target.unsqueeze(1).float()
    loss = loss(out, target)*(1 + addloss)
    # loss = torch.tensor([torch.nan])
    if torch.isnan(loss.detach()):
        losses = -frac*(target*torch.log(torch.nn.functional.sigmoid(out))+(1-target)*torch.log(1-torch.nn.functional.sigmoid(out)))
        print("-frac*[target*log(σ(out))+(1-target)⋅log(1-σ(out​))] = ", losses)
        nan_index = losses.isnan()
        print("causes of nan losses: ")
        print("- target: ", target[nan_index])
        print("- out: ", out[nan_index])
        # with open("losses.csv", 'w') as f:
        #     for loss in -frac*(target*torch.log(torch.nn.functional.sigmoid(out))+(1-target)*torch.log(1-torch.nn.functional.sigmoid(out))):
        #         f.write("{},\n".format(loss.item()))
        print("out: ", out)
        print("target: ", target)
        # print("bi_target: ", bi_target)
        print("pred: ", pred)
        print("true_positives: ", true_positives)
        print("true_negatives: ", true_negatives)
        print("frac, rev_frac: ", frac, ", ", rev_frac)
        print("addloss: ", addloss)
        print("loss: ", loss)
        raise Exception("loss is not a number")
    # assert not torch.isnan(loss.detach()), f"out: {out}\ndata: {target}\nLoss: {loss}\nWeight: {frac}"
    return loss, true_positives, true_negatives

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
        pred = model(data).argmax(dim=-1)
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