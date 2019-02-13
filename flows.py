import numpy as np
import sys
import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from MADE import MADE
from common import FeedForwardNet

# Transformation functions

class Affine():
    num_params = 2

    @staticmethod
    def get_pseudo_params(nn_outp):
        a = nn_outp[..., 0] # [B, D]
        var_outp = nn_outp[..., 1]
        
        b = torch.exp(0.5*var_outp)
        logbsq = var_outp

        return a, logbsq, b

    @staticmethod
    def standard(x, nn_outp):
        a, logbsq, b = Affine.get_pseudo_params(nn_outp)

        y = a + b*x
        logdet = 0.5*logbsq.sum(-1)
        
        return y, logdet

    @staticmethod
    def reverse(y, nn_outp):
        a, logbsq, b = Affine.get_pseudo_params(nn_outp)

        x = (y - a)/b
        logdet = 0.5*logbsq.sum(-1)

        return x, logdet

def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2)-1))

def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2)+1))

class NLSq():
    num_params = 5
    logA = math.log(8*math.sqrt(3)/9-0.05) # 0.05 is a small number to prevent exactly 0 slope

    @staticmethod
    def get_pseudo_params(nn_outp):
        a = nn_outp[..., 0] # [B, D]
        logb = nn_outp[..., 1]*0.4
        B = nn_outp[..., 2]*0.3
        logd = nn_outp[..., 3]*0.4
        f = nn_outp[..., 4]

        b = torch.exp(logb)
        d = torch.exp(logd)
        c = torch.tanh(B)*torch.exp(NLSq.logA + logb - logd)

        return a, b, c, d, f
    
    @staticmethod
    def standard(x, nn_outp):
        a, b, c, d, f = NLSq.get_pseudo_params(nn_outp)
        
        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        f = f.double()
        x = x.double()

        aa = -b*d.pow(2)
        bb = (x-a)*d.pow(2) - 2*b*d*f
        cc = (x-a)*2*d*f - b*(1+f.pow(2))
        dd = (x-a)*(1+f.pow(2)) - c

        p = (3*aa*cc - bb.pow(2))/(3*aa.pow(2))
        q = (2*bb.pow(3) - 9*aa*bb*cc + 27*aa.pow(2)*dd)/(27*aa.pow(3))
        
        t = -2*torch.abs(q)/q*torch.sqrt(torch.abs(p)/3)
        inter_term1 = -3*torch.abs(q)/(2*p)*torch.sqrt(3/torch.abs(p))
        inter_term2 = 1/3*arccosh(torch.abs(inter_term1-1)+1)
        t = t*torch.cosh(inter_term2)

        tpos = -2*torch.sqrt(torch.abs(p)/3)
        inter_term1 = 3*q/(2*p)*torch.sqrt(3/torch.abs(p))
        inter_term2 = 1/3*arcsinh(inter_term1)
        tpos = tpos*torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        y = t - bb/(3*aa)

        arg = d*y + f
        denom = 1 + arg.pow(2)

        x_new = a + b*y + c/denom

        logdet = -torch.log(b - 2*c*d*arg/denom.pow(2)).sum(-1)

        y = y.float()
        logdet = logdet.float()

        return y, logdet


    @staticmethod
    def reverse(y, nn_outp):
        a, b, c, d, f = NLSq.get_pseudo_params(nn_outp)

        arg = d*y + f
        denom = 1 + arg.pow(2)
        x = a + b*y + c/denom

        logdet = -torch.log(b - 2*c*d*arg/denom.pow(2)).sum(-1)

        return x, logdet
 
class SCFLayer(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function, hidden_order=None, swap_trngen_dirs=False,
                 input_order=None, conditional_inp_dim=None, dropout=[0, 0]):
        super().__init__()

        self.net = FeedForwardNet(data_dim//2 + conditional_inp_dim, n_hidden_units, (data_dim-(data_dim//2))*transform_function.num_params, n_hidden_layers, nonlinearity, dropout=dropout[1])

        self.train_func = transform_function.standard if swap_trngen_dirs else transform_function.reverse
        self.gen_func = transform_function.reverse if swap_trngen_dirs else transform_function.standard
        self.input_order = input_order
        
        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """

        data_dim = len(self.input_order)
        assert data_dim == inputs[0].shape[-1]

        first_indices = torch.arange(len(self.input_order))[self.input_order <= data_dim//2] # This is <= because input_order goes from 1 to data_dim+1
        second_indices = torch.arange(len(self.input_order))[self.input_order > data_dim//2]

        if self.use_cond_inp:
            y, logdet, cond_inp = inputs
            net_inp = torch.cat([y[..., first_indices], cond_inp], -1)
        else:
            y, logdet = inputs
            net_inp = y[..., first_indices]

        nn_outp = self.net(net_inp).view(*net_inp.shape[:-1], data_dim-(data_dim//2), -1) # [..., ~data_dim/2, num_params]

        x = torch.tensor(y)
        x[..., second_indices], change_logdet = self.train_func(y[..., second_indices], nn_outp)

        return x, logdet + change_logdet, cond_inp
    
    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """

        data_dim = len(self.input_order)
        assert data_dim == inputs[0].shape[-1]

        first_indices = torch.arange(len(self.input_order))[self.input_order <= data_dim//2] # This is <= because input_order goes from 1 to data_dim+1
        second_indices = torch.arange(len(self.input_order))[self.input_order > data_dim//2]

        if self.use_cond_inp:
            x, logdet, cond_inp = inputs
            net_inp = torch.cat([x[..., first_indices], cond_inp], -1)
        else:
            x, logdet = inputs
            net_inp = x[..., first_indices]

        nn_outp = self.net(net_inp).view(*net_inp.shape[:-1], data_dim-(data_dim//2), -1) # [..., ~data_dim/2, num_params]

        y = torch.tensor(x)
        y[..., second_indices], change_logdet = self.gen_func(x[..., second_indices], nn_outp)

        return y, logdet + change_logdet, cond_inp


class AFLayer(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function, hidden_order='sequential', swap_trngen_dirs=False,
                 input_order=None, conditional_inp_dim=None, dropout=[0, 0], coupling_level=0):
        super().__init__()

        self.made = MADE(data_dim, n_hidden_layers, n_hidden_units, nonlinearity, hidden_order,
                         out_dim_per_inp_dim=transform_function.num_params, input_order=input_order, conditional_inp_dim=conditional_inp_dim,
                         dropout=dropout)

        self.train_func = transform_function.standard if swap_trngen_dirs else transform_function.reverse
        self.gen_func = transform_function.reverse if swap_trngen_dirs else transform_function.standard
        self.output_order = self.made.end_order
        self.data_dim = data_dim
        
        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """

        if self.use_cond_inp:
            y, logdet, cond_inp = inputs
            nn_outp = self.made([y, cond_inp]) # [B, D, 2]
        else:
            y, logdet = inputs
            nn_outp = self.made(y) # [B, D, 2]

        x, change_logdet = self.train_func(y, nn_outp)

        return x, logdet + change_logdet, cond_inp

    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """
        if self.use_cond_inp:
            x, logdet, cond_inp = inputs
        else:
            x, logdet = inputs

        y = torch.tensor(x)
        for idx in range(self.data_dim):
            t = (self.output_order==idx).nonzero()[0][0]

            if self.use_cond_inp:
                nn_outp = self.made([y, cond_inp])
            else:
                nn_outp = self.made(y)

            y[..., t:t+1], new_partial_logdet = self.gen_func(x[..., t:t+1], nn_outp[..., t:t+1, :])
            logdet += new_partial_logdet

        return y, logdet, cond_inp

# Full flow combining multiple layers

class Flow(nn.Module):
    def __init__(self, data_dim, n_hidden_layers, n_hidden_units, nonlinearity, num_flow_layers, transform_function,
                 iaf_like=False, hidden_order='sequential',
                 swap_trngen_dirs=False, conditional_inp_dim=None, dropout=[0, 0], reverse_between_layers=True,
                 scf_layers=False, reverse_first_layer=False):
        super().__init__()

        if transform_function == 'affine':
            transform_function = Affine
        elif transform_function == 'nlsq':
            transform_function = NLSq
        elif transform_function != Affine and transform_function != NLSq: # Can pass string or actual class
            raise NotImplementedError('Only the affine transformation function has been implemented')

        if scf_layers:
            AutoregressiveLayer = SCFLayer
        else:
            AutoregressiveLayer = AFLayer

        # Note: This ordering is the ordering as applied to go from data -> base
        flow_layers = []
    
        input_order = torch.arange(data_dim)+1

        if reverse_first_layer:
            input_order = reversed(input_order)

        for i in range(num_flow_layers):
            flow_layers.append(AutoregressiveLayer(data_dim, n_hidden_layers, n_hidden_units, nonlinearity, transform_function,
                               hidden_order=hidden_order, swap_trngen_dirs=swap_trngen_dirs, input_order=input_order,
                               conditional_inp_dim=conditional_inp_dim, dropout=dropout))
            if reverse_between_layers:
                input_order = reversed(input_order)

        self.flow = nn.Sequential(*flow_layers)
        self.use_cond_inp = conditional_inp_dim is not None

    def forward(self, inputs):
        """
            Defines the reverse pass which is used during training
            logdet means log det del_y/del_x
        """
        if self.use_cond_inp:
            y, cond_inp = inputs
        else:
            y = inputs

        logdet = torch.zeros(y.shape[:-1], device=y.device)

        if self.use_cond_inp:
            x, logdet, _ = self.flow([y, logdet, cond_inp])
        else:
            x, logdet = self.flow([y, logdet])

        return x, logdet

    def generate(self, inputs):
        """
            Defines the forward pass which is used during testing
            logdet means log det del_y/del_x
        """

        if self.use_cond_inp:
            x, cond_inp = inputs
        else:
            x = inputs

        logdet = torch.zeros(x.shape[:-1], device=x.device)
        y = x
        for flow_layer in reversed(self.flow):
            if self.use_cond_inp:
                y, logdet, _ = flow_layer.generate([y, logdet, cond_inp])
            else:
                y, logdet = flow_layer.generate([y, logdet])

        return y, logdet
