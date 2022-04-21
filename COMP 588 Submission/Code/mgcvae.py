import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rdkit import RDLogger
from rdkit import Chem
RDLogger.DisableLog('rdApp.*')
import warnings
import numpy as np
warnings.filterwarnings(action='ignore')
import argparse


class CVAE(nn.Module):

    # added num_conditions parameter and used it to properly scale the input matrix
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, c_dim, num_conditions):
        super(CVAE, self).__init__()

        self.fc1 = nn.Linear(x_dim + c_dim * num_conditions, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

        # decoder part
        self.fc4 = nn.Linear(c_dim * num_conditions + z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x, condition_list):

        # dynamically adds conditions instead of having a static set. the same is done in the decoder
        input_matrix = [x]
        for cond in condition_list:
            input_matrix.append(cond)

        concat_input = torch.cat(input_matrix, 1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))

        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add(mu)

    def decoder(self, z, condition_list):

        input_matrix = [z]
        for cond in condition_list:
            input_matrix.append(cond)
        concat_input = torch.cat(input_matrix, 1)

        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))

        return torch.sigmoid(self.fc6(h))

    def forward(self, x, condition_list, out_dim):
        mu, log_var = self.encoder(x.view(-1, out_dim), condition_list)

        # This would be where the monotonic reparamaterization would occur, but unfortunately it wasn't successful
        # or even interestingly incorrect

        z = self.sampling(mu, log_var)

        return self.decoder(z, condition_list), mu, log_var


def loss_function(recon_x, x, mu, log_var, out_dim):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, out_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD


def one_hot(labels, class_size, zero_value):

    # added the zero_value parameter to allow for a normalized mean adjustment
    targets = torch.zeros(labels.shape[0], class_size)
    for i, label in enumerate(labels):
        index = round(label.item()) + zero_value
        targets[i, index] = 1

    return Variable(targets)

