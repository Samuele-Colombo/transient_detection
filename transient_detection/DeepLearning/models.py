# DeepLearning/models.py
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
Module containing implementations of graph convolutional networks (GCN) for classification.

The GCNClassifier class extends the torch.nn.Module class and implements a GCN model with multiple layers
and a linear output layer. The forward pass applies a series of GCNConv layers and an activation function
to the input data, followed by a linear layer that produces the output. Dropout is applied between each GCNConv
layer to regularize the model.

Example
-------

To use the GCNClassifier, you need to first create an instance of the class and pass the necessary arguments
to the constructor. Then, you can use the model to make predictions by calling the `forward` method.

>>> model = GCNClassifier(num_layers=2, input_dim=10, hidden_dim=16, output_dim=5)
>>> output = model(x, edge_index)

"""

import torch
from torch_geometric.nn import GCNConv
from transient_detection.DeepLearning.utilities import gaussian_kernel

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation_function=torch.nn.functional.relu, h=1):
        """
        Initialize the GCNClassifier model.

        Parameters
        ----------
        num_layers : int
            The number of GCN layers to use in the model.
        input_dim : int
            The input dimension of the data.
        hidden_dim : int
            The hidden dimension of the GCN layers.
        output_dim : int
            The output dimension of the model (i.e. the number of classes).
        activation_function : function, optional
            The activation function of each layer. Default is `torch.nn.functional.relu`.

        """
        super(GCNClassifier, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
        self.activation_function = activation_function
        self.h = h

    def forward(self, x, edge_index, edge_attr, dropout_rate=0.5):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data, with shape (batch_size, input_dim).
        edge_index : torch.Tensor
            The edge indices, with shape (2, num_edges).
        dropout_rate : float, optional
            The dropout rate. Default is 0.5.

        Returns
        -------
        torch.Tensor
            The model's output, with shape (batch_size, output_dim).
        
        """
        # print("edge_index: ", edge_index)
        # GB = 1024 * 1024 * 1024
        # print("size of x: ", x.element_size() * x.nelement() / GB, "GB")
        # print("0: ", torch.cuda.memory_allocated() / GB, "GB")
        for i, conv in enumerate(self.convs):
            # print(i, ":0: ", x)
            # print(i, ":1: ", torch.cuda.memory_allocated() / GB, "GB")
            x = conv(x, edge_index, edge_attr)
            # print(i, ":1: ", x)
            # print(i, ":2: ", torch.cuda.memory_allocated() / GB, "GB")
            x = self.activation_function(x)
            # print(i, ":2: ", x)
            # print(i, ":3: ", torch.cuda.memory_allocated() / GB, "GB")
            x = torch.nn.functional.dropout(x, p=dropout_rate, training=self.training)
        # print("end: ", x)
        # print("end: ", torch.cuda.memory_allocated() / GB, "GB")
        x = self.lin(x)
        # print("result: ", x)
        # print("result: ", torch.cuda.memory_allocated() / GB, "GB")
        return torch.nn.functional.sigmoid(x)

# class Net(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, activation_function=F.relu):
#         super().__init__()

#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers - 1):
#             self.convs.append(GCNConv(in_channels, hidden_channels))
        
#         self.activation_function = activation_function

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = self.activation_function(x)
#             x = F.dropout(x, training=self.training)
        
#         return F.log_softmax(x, dim=1)