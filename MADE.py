import math
import collections
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class MaskedHiddenLayer(nn.Module):
    def __init__(self, d_in, d_out, data_dim, nonlinearity, previous_m_k, output_order, bias=True, alt_init=False):
        super().__init__()
        if nonlinearity == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif nonlinearity == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlinearity == 'elu':
            self.nonlin = nn.ELU(inplace=True)
        elif nonlinearity == None:
            self.nonlin = lambda x : x
        else:
            raise NotImplementedError('only relu, tanh, and elu nonlinearities have been implemented')

        self.weight = Parameter(torch.Tensor(d_out, d_in))
        if bias:
            self.bias = Parameter(torch.Tensor(d_out))
        else:
            self.register_parameter('bias', None)

        self.alt_init = alt_init
        self.reset_parameters()

        if isinstance(output_order, str):
            if output_order == 'random':
                self.m_k = torch.empty(d_out, dtype=torch.long).random_(1, data_dim)
            elif output_order == 'sequential':
                self.m_k = torch.arange(0, data_dim)
                self.m_k = self.m_k.repeat(d_out//data_dim+1)[:d_out]
        else:
            # Allow for the network to produce multiple outputs conditioned on the same degree
            self.m_k = output_order.repeat(d_out//data_dim)

        mask = (self.m_k[:, None] >= previous_m_k[None, :]).float()
        self.register_buffer('mask', mask)

    def reset_parameters(self):
        if self.alt_init:
            stdv = 1. / math.sqrt(self.weight.size(0) + 1)
            self.weight.data.uniform_(-0.001, 0.001)
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            if self.alt_init:
                self.bias.data.zero_()
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = F.linear(x, Variable(self.mask)*self.weight, self.bias)
        x = self.nonlin(x)

        return x


class MADE(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, hidden_order, bias=True, out_dim_per_inp_dim=1, input_order=None,
                 conditional_inp_dim=None, dropout=[0, 0], nonar=False, alt_init=True):
        super().__init__()

        if not isinstance(dropout, collections.Iterable) or len(dropout) != 2:
            raise ValueError('dropout argument should be an iterable with [input drop fraction, hidden drop fraction')

        layers = []
        if input_order is None:
            previous_m_k = torch.arange(data_dim)+1
            end_order = torch.arange(data_dim)
        else:
            if not nonar and not np.all(np.sort(input_order) == np.arange(data_dim)+1):
                raise ValueError('input_order must contain 1 through data_dim, inclusive, in any order')
            previous_m_k = input_order
            end_order = input_order-1

        if conditional_inp_dim is not None:
            previous_m_k = torch.cat([previous_m_k, torch.zeros(conditional_inp_dim, dtype=previous_m_k.dtype)])

        effective_data_dim = torch.max(previous_m_k) # This is only used to set the m_k values for each hidden layer

        for i in range(n_hidden_layers):
            if i == 0:
                d_in = data_dim
                if conditional_inp_dim is not None:
                    d_in += conditional_inp_dim
                drop_val = dropout[0]
            else:
                d_in = n_hidden_units
                drop_val = dropout[1]

            if drop_val > 0:
                layers.append(nn.Dropout(drop_val))

            new_layer = MaskedHiddenLayer(d_in, n_hidden_units, effective_data_dim, nonlinearity, previous_m_k, hidden_order, bias=bias, alt_init=alt_init)
            previous_m_k = new_layer.m_k
            layers.append(new_layer)

        layers.append(MaskedHiddenLayer(n_hidden_units, data_dim*out_dim_per_inp_dim, data_dim, None, previous_m_k, end_order, bias=bias))

        self.network = nn.Sequential(*layers)
        self.data_dim = data_dim
        self.out_dim_per_inp_dim = out_dim_per_inp_dim
        self.end_order = end_order
        self.conditional_inp_dim = conditional_inp_dim

    def forward(self, inputs):
        if self.conditional_inp_dim is not None:
            x, cond_inp = inputs
            x = torch.cat([x, cond_inp], -1)
        else:
            x = inputs

        x = self.network(x)

        if self.out_dim_per_inp_dim == 1:
            return x

        # If the network produces multiple outputs conditioned on the same degree, return as [B, data_dim, out_dim_per_inp_dim]
        #x = torch.transpose(x.view(-1, self.out_dim_per_inp_dim, self.data_dim), -1, -2)
        #if x_1d:
        #    x = x.squeeze()
        x = x.view(*x.shape[:-1], self.out_dim_per_inp_dim, self.data_dim)
        x = torch.transpose(x, -1, -2)

        return x

