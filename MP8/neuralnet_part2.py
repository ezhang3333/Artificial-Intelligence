# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP8 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Activation
        self.act = nn.LeakyReLU()
        # Compute flattened feature size after two poolings:
        # 31x31 -> pool -> 15x15 -> pool -> 7x7
        flat_dim = 32 * 7 * 7
        # Fully connected head
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, out_size)
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)


    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """

        N = x.shape[0]
        # reshape to (N,3,31,31)
        x = x.view(N, 3, 31, 31)
        # conv1 -> act -> pool
        x = self.pool(self.act(self.conv1(x)))  # (N,16,15,15)
        # conv2 -> act -> pool
        x = self.pool(self.act(self.conv2(x)))  # (N,32,7,7)
        # flatten
        x = x.view(N, -1)                       # (N,32*7*7)
        # fc1 -> act
        x = self.act(self.fc1(x))               # (N,128)
        # fc2 (raw logits)
        logits = self.fc2(x)                    # (N,out_size)
        return logits

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        self.optimizer.zero_grad()
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """


    # Standardize features as in Part1
    train_mean = torch.mean(train_set, dim=0)
    train_std = torch.std(train_set, dim=0)
    train_std[train_std == 0] = 1.0
    train_norm = (train_set - train_mean) / train_std
    dev_norm = (dev_set - train_mean) / train_std

    # Prepare data loader
    X_train = train_norm.numpy()
    y_train = train_labels.numpy()
    train_dataset = get_dataset_from_arrays(X_train, y_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    # Determine sizes
    in_size = train_set.shape[1]  # 2883
    out_size = int(torch.max(train_labels).item()) + 1
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(lrate=0.001, loss_fn=loss_fn,
                    in_size=in_size, out_size=out_size)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        net.train()
        for batch in train_loader:
            # extract features and labels
            if isinstance(batch, dict):
                x_batch = batch.get('features', batch.get('x'))
                y_batch = batch.get('labels', batch.get('y'))
            else:
                x_batch, y_batch = batch
            epoch_loss += net.step(x_batch, y_batch)
        losses.append(epoch_loss / len(train_loader))

    # Evaluation
    net.eval()
    with torch.no_grad():
        scores = net(dev_norm)
        _, preds = torch.max(scores, dim=1)
    predicted = preds.numpy().astype(np.int64)
    return losses, predicted, net

