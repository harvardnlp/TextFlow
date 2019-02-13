import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

import math
import sys

from lstm_flow import AFPrior
from common import FeedForwardNet
from utils import make_pos_cond

class InferenceBlock(nn.Module):
    def __init__(self, inf_inp_dim, hidden_size, zsize, dropout_p, q_rnn_layers, dropout_locations, max_T):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        rnn_q_inp_size = inf_inp_dim + 2*max_T
        self.rnn_q = torch.nn.LSTM(rnn_q_inp_size, hidden_size, q_rnn_layers, dropout=dropout_p if 'rnn_x' in dropout_locations else 0, bidirectional=True)
        
        self.q_base_ff = nn.Linear(hidden_size*2, zsize*2)

        self.hidden_size = hidden_size
        self.zsize = zsize
        self.q_rnn_layers = q_rnn_layers
        self.dropout_locations = dropout_locations
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.07
        self.q_base_ff.weight.data.uniform_(-init_range, init_range)
        self.q_base_ff.bias.data.zero_()

    def sample_q_z(self, inf_inp, lengths, cond_inp, ELBO_samples):
        """
            output is z [T, B, s, E]
        """

        ## Run RNN over input
        T, B = inf_inp.shape[:2]
        hidden_rnn = self.init_hidden_rnn(B)

        inf_inp_packed = torch.cat((inf_inp, cond_inp), -1)

        total_length = inf_inp_packed.shape[0]
        inf_inp_packed = torch.nn.utils.rnn.pack_padded_sequence(inf_inp_packed, lengths)
        rnn_outp, _ = self.rnn_q(inf_inp_packed, hidden_rnn) # [T, B, hidden_size], [num_layers, B, hidden_size]x2
        rnn_outp = torch.nn.utils.rnn.pad_packed_sequence(rnn_outp, total_length=total_length)[0]
        
        if 'rnn_x' in self.dropout_locations:
            rnn_outp = self.dropout(rnn_outp)
        
        ## Sample ELBO_sample z's from RNN output
        rnn_outp = rnn_outp[:, :, None, :].repeat(1, 1, ELBO_samples, 1)
        q_z_base = self.q_base_ff(rnn_outp)

        q_z_base = q_z_base.view(*rnn_outp.shape[:-1], self.zsize, 2)
        z_base_mean = q_z_base[..., 0]
        z_base_logvar = q_z_base[..., 1]
        z_base_std = torch.exp(0.5*z_base_logvar)

        eps_initial = torch.randn(T, B, ELBO_samples, self.zsize, device=z_base_mean.device)
        z = z_base_mean + z_base_std*eps_initial # [T, B, s, E]

        log_q_z = -1/2*(math.log(2*math.pi) + z_base_logvar + (z - z_base_mean).pow(2)/z_base_std.pow(2)).sum(-1) # [T, B, s]
        
        # Reshape z into B and s
        z = z.view(T, B, ELBO_samples, self.zsize) # [T, B, s, E]

        return z, log_q_z

    def init_hidden_rnn(self, batch_size):
        weight = next(self.parameters())
        h = weight.new_zeros(self.q_rnn_layers*2, batch_size, self.hidden_size)
        c = weight.new_zeros(self.q_rnn_layers*2, batch_size, self.hidden_size)
        return (h, c)


class GenerativeBlock(nn.Module):
    def __init__(self, hidden_size, zsize, prior_type, dropout_p, dropout_locations, outp_rnn_layers, max_T, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Prior
        if prior_type not in ['AF', 'IAF', 'hiddenflow_only']:
            raise ValueError('Error, prior_type %s unknown' % prior_type)

        p_rnn_layers = kwargs['p_rnn_layers']
        p_rnn_units = kwargs['p_rnn_units']
        if p_rnn_units < 0:
            p_rnn_units = hidden_size

        p_num_flow_layers = kwargs['p_num_flow_layers']
        transform_function = kwargs['transform_function']
        hiddenflow_params = {k: v for k, v in kwargs.items() if 'hiddenflow' in k}

        self.prior = AFPrior(p_rnn_units, zsize, dropout_p, dropout_locations, prior_type, p_num_flow_layers, p_rnn_layers, 
                             max_T=max_T, transform_function=transform_function, hiddenflow_params=hiddenflow_params)

        # BiLSTM
        self.rnn_outp = nn.LSTM(zsize + 2*max_T, hidden_size, outp_rnn_layers, dropout=dropout_p if 'rnn_outp' in dropout_locations else 0, bidirectional=True)
        self.outp_dim = 2*hidden_size + zsize

        self.outp_rnn_layers = outp_rnn_layers
        self.hidden_size = hidden_size
        self.zsize = zsize
        self.dropout_locations = dropout_locations

    def apply_bilstm(self, z, lengths_s, cond_inp_s):
        """
            z is [T, B, s, E]
        """
        T, B, ELBO_samples = z.shape[:3]

        hidden_outp = self.init_hidden(B)
        hidden_outp = tuple(h[:, :, None, :].repeat(1, 1, ELBO_samples, 1).view(-1, B*ELBO_samples, self.hidden_size) for h in hidden_outp)

        z = z.view(T, B*ELBO_samples, self.zsize)
        z_packed = z
        if 'z_before_outp' in self.dropout_locations:
            z_packed = self.dropout(z_packed)

        z_packed = torch.cat((z_packed, cond_inp_s), -1)
        
        total_length = z_packed.shape[0]
        z_packed = nn.utils.rnn.pack_padded_sequence(z_packed, lengths_s)
        rnn_outp_outp, _ = self.rnn_outp(z_packed, hidden_outp)
        rnn_outp_outp = nn.utils.rnn.pad_packed_sequence(rnn_outp_outp, total_length=total_length)[0]

        if 'rnn_outp' in self.dropout_locations:
            rnn_outp_outp = self.dropout(rnn_outp_outp)

        z_cat = z.view(T, B, ELBO_samples, self.zsize)
        rnn_outp_outp = rnn_outp_outp.view(T, B, ELBO_samples, 2, self.hidden_size)

        # Reorganize rnn output
        hidden_outp_sep = hidden_outp[0].view(self.outp_rnn_layers, 2, B, ELBO_samples, self.hidden_size)
            
        rnn_outp_outp_shifted_forward = torch.cat((hidden_outp_sep[-1:, 0], rnn_outp_outp[:, :, :, 0]), 0)[:-1] # [T, B, s, hidden]
        rnn_outp_outp_shifted_backward = torch.cat((rnn_outp_outp[:, :, :, 1], hidden_outp_sep[-1:, 1]), 0)[1:]
        
        z_with_hist = torch.cat((z_cat, rnn_outp_outp_shifted_forward, rnn_outp_outp_shifted_backward), -1)

        return z_with_hist
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h = weight.new_zeros(self.outp_rnn_layers*2, batch_size, self.hidden_size)
        c = weight.new_zeros(self.outp_rnn_layers*2, batch_size, self.hidden_size)
        return (h, c)
    

class DFModel(nn.Module):
    def __init__(self, vocab_size, loss_weights, n_inp_embedding, hidden_size, zsize, dropout_p, dropout_locations, # general parameters
                 prior_type, gen_bilstm_layers, prior_kwargs, # gen block parameters
                 q_rnn_layers, tie_weights, max_T, indep_bernoulli=False): # misc parameters
        super().__init__()

        for loc in dropout_locations:
            if loc not in ['embedding', 'rnn_x', 'z_before_prior', 'prior_rnn_inp', 'prior_rnn', 'prior_ff', 'z_before_outp', 'rnn_outp', 'outp_ff']:
                raise ValueError('dropout location %s not a valid location' % loc)

        self.dropout = torch.nn.Dropout(dropout_p)

        ## Initial embedding
        self.input_embedding = torch.nn.Embedding(vocab_size, n_inp_embedding)

        ## Latent models
        self.generative_model = GenerativeBlock(hidden_size, zsize, prior_type, dropout_p, dropout_locations, gen_bilstm_layers, max_T, **prior_kwargs)

        self.inference_model = InferenceBlock(n_inp_embedding, hidden_size, zsize, dropout_p, q_rnn_layers, dropout_locations, max_T)

        ## Generative output to x
        self.outp_ff = FeedForwardNet(self.generative_model.outp_dim, hidden_size, vocab_size, 1, 'none', dropout=dropout_p if 'outp_ff' in dropout_locations else 0)
        
        if tie_weights:
            self.outp_ff.network[-1].weight = self.input_embedding.weight

        if indep_bernoulli:
            self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(loss_weights, reduction='none')

        self.indep_bernoulli = indep_bernoulli
        self.vocab_size = vocab_size
        self.dropout_locations = dropout_locations
        self.max_T = max_T

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.07
        self.input_embedding.weight.data.uniform_(-init_range, init_range)
        
    def generate(self, lengths, temp=1.0, argmax_x=True):
        """
            lengths is [B] with lengths of each sentence in the batch
            all inputs should be on the same compute device
        """

        T = torch.max(lengths)
        B = lengths.shape[0]

        ## Calculate position conditioning
        pos_cond = make_pos_cond(T, B, lengths, self.max_T)

        ## Generate z's from prior
        z, _ = self.generative_model.prior.generate(lengths, cond_inp=pos_cond, temp=temp)
        z = z[:, :, None, :]

        ## Apply BiLSTM part of likelihood
        gen_outp = self.generative_model.apply_bilstm(z, lengths, cond_inp_s=pos_cond)
        gen_outp = gen_outp.squeeze(2)
        
        ## Final output
        scores = self.outp_ff(gen_outp) # [T, B, V]

        if argmax_x:
            if self.indep_bernoulli:
                probs = torch.sigmoid(scores)
                generation = (probs > 0.5).long()
            else:
                generation = torch.argmax(scores, -1)
        else:
            if self.indep_bernoulli:
                word_dist = Bernoulli(logits=scores)
            else:
                word_dist = Categorical(logits=scores)
            generation = word_dist.sample()

        return generation

    def evaluate_x(self, x, lengths, ELBO_samples=1):
        """
            x is [T, B] with indices of tokens
            lengths is [B] with lengths of each sentence in the batch
            all inputs should be on the same compute device
        """

        T, B = x.shape[:2]
        
        ## Create ELBO_sample versions of inputs copied across a new dimension
        lengths_s = lengths[:, None].repeat(1, ELBO_samples).view(-1)

        pos_cond = make_pos_cond(T, B, lengths, self.max_T)
        pos_cond_s = pos_cond[:, :, None, :].repeat(1, 1, ELBO_samples, 1).view(T, B*ELBO_samples, self.max_T*2)

        ## Get the initial x embeddings
        if self.indep_bernoulli:
            embeddings = torch.matmul(x, self.input_embedding.weight)
        else:
            embeddings = self.input_embedding(x) # [T, B, n_inp_embedding]

        if 'embedding' in self.dropout_locations:
            embeddings = self.dropout(embeddings)

        z, log_q_z = self.inference_model.sample_q_z(embeddings, lengths, pos_cond, ELBO_samples) # [T, B, s, E]
            
        log_p_z = self.generative_model.prior.evaluate(z, lengths_s, cond_inp_s=pos_cond_s)
        gen_outp = self.generative_model.apply_bilstm(z, lengths_s, cond_inp_s=pos_cond_s)
        gen_outp = gen_outp.view(gen_outp.shape[0], B*ELBO_samples, gen_outp.shape[-1]) # [T, B*s, 2*hidden]

        ## Final output
        scores = self.outp_ff(gen_outp) # [T, B, s, V]

        if self.indep_bernoulli:
            targets = x[:, :, None, :].repeat(1, 1, ELBO_samples, 1)
            reconst_loss = self.cross_entropy(scores.view(-1, self.vocab_size), targets.view(-1, self.vocab_size)).view(T, B, ELBO_samples, self.vocab_size).sum(-1) # [T, B, s]
        else:
            targets = x[:, :, None].repeat(1, 1, ELBO_samples)
            reconst_loss = self.cross_entropy(scores.view(-1, self.vocab_size), targets.view(-1)).view(T, B, ELBO_samples)

        kl_loss = (log_q_z - log_p_z) # [T, B, s]
        
        return reconst_loss, kl_loss
