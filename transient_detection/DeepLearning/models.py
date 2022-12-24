# DeepLearning/models.py
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

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation_function=torch.nn.functional.relu):
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
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = torch.nn.Linear(hidden_dim, output_dim)
        self.activation_function = activation_function

    def forward(self, x, edge_index, dropout_rate=0.5):
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
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation_function(x)
            x = torch.nn.functional.dropout(x, p=dropout_rate, training=self.training)
        x = self.lin(x)
        return x

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