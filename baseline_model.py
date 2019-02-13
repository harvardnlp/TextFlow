import time

import torch
from torch import nn
from torch.distributions import Categorical, Bernoulli

from common import FeedForwardNet
from utils import make_pos_cond

class LSTMModel(nn.Module):

    def __init__(self, vocab_size, loss_weights, n_inp_units, n_hidden_units, n_layers, dropout_p, T_condition=False, max_T=-1, tie_weights=False, indep_bernoulli=False):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.input_embedding = nn.Embedding(vocab_size, n_inp_units)

        rnn_inp_size = n_inp_units
        if T_condition:
            rnn_inp_size += max_T*2

        self.rnn = nn.LSTM(rnn_inp_size, n_hidden_units, n_layers, dropout=dropout_p)

        self.output_embedding = FeedForwardNet(n_hidden_units, n_inp_units, vocab_size, 1, 'none')

        if tie_weights:
            self.output_embedding.network[-1].weight = self.input_embedding.weight

        self.indep_bernoulli = indep_bernoulli
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.T_condition = T_condition
        self.max_T = max_T
        
        if self.indep_bernoulli:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(loss_weights, reduction='none')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_embedding.weight)

    def forward(self, x, lengths):
        # Input is [T, B] with index of word
        T, B = x.shape[0], x.shape[1]

        hidden = self.init_hidden(B)

        if self.T_condition:
            cond_inp = make_pos_cond(T, B, lengths.cpu(), self.max_T).to(x.device)
        
        if self.indep_bernoulli:
            embeddings = torch.matmul(x, self.input_embedding.weight)
        else:
            embeddings = self.input_embedding(x)
        embeddings = self.dropout(embeddings) # [T, B, n_inp_units]

        if self.T_condition:
            cond_inp_shifted = torch.cat((cond_inp[1:], torch.zeros((1, B, self.max_T*2), device=cond_inp.device)), 0)
            embeddings = torch.cat((embeddings, cond_inp_shifted), -1)
        
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths)
        rnn_outp, _ = self.rnn(embeddings, hidden) # [T, B, n_hidden_units], [num_layers, B, n_hidden_units]x2
        rnn_outp = nn.utils.rnn.pad_packed_sequence(rnn_outp)[0]

        rnn_outp = torch.cat((hidden[0][-1:], rnn_outp), 0)[:-1]
        rnn_outp = self.dropout(rnn_outp)

        scores = self.output_embedding(rnn_outp) # [T, B, V]

        if self.indep_bernoulli:
            loss = self.criterion(scores.view(-1, scores.shape[-1]), x.view(-1, x.shape[-1])).view(scores.shape).sum(-1)
            # This doesn't 0 out loss values from padding, but later on the main loop will do that
        else:
            loss = self.criterion(scores.view(-1, scores.shape[-1]), x.view(-1)).view(scores.shape[:-1]) # [T, B]

        return loss

    def generate(self, T, B):
        if not self.T_condition:
            raise NotImplementedError("Only the version conditioned on T has been implemented.")

        hidden = self.init_hidden(B)
        lengths = torch.tensor([T]*B)
        device = hidden[0].device

        cond_inp = make_pos_cond(T, B, lengths, self.max_T).to(device)

        if self.indep_bernoulli:
            generation = torch.zeros(T, B, self.vocab_size, dtype=torch.long, device=device)
        else:
            generation = torch.zeros(T, B, dtype=torch.long, device=device)

        last_rnn_outp = hidden[0][-1]
        for t in range(T):
            scores = self.output_embedding(last_rnn_outp) # [B, V]
            if self.indep_bernoulli:
                word_dist = Bernoulli(logits=scores)
            else:
                word_dist = Categorical(logits=scores)

            selected_index = word_dist.sample()
            generation[t] = selected_index

            if t < T-1:
                if self.indep_bernoulli:
                    inp_embeddings = torch.matmul(generation[t].float(), self.input_embedding.weight)
                else:
                    inp_embeddings = self.input_embedding(generation[t]) # [B, E]
                inp_embeddings = torch.cat((inp_embeddings, cond_inp[t+1]), -1)

                last_rnn_outp, hidden = self.rnn(inp_embeddings[None, :, :], hidden)
                last_rnn_outp = last_rnn_outp[0]

        return generation

    def gen_one_noTcond(self, eos_index, max_T):
        hidden = self.init_hidden(1)
        device = hidden[0].device
        
        last_rnn_outp = hidden[0][-1] # [1, C]
        generation = []

        for t in range(max_T):
            scores = self.output_embedding(last_rnn_outp) # [1, V]
            word_dist = Categorical(logits=scores)
            selected_index = word_dist.sample() # [1]

            if selected_index == eos_index:
                break

            generation.append(selected_index)
            inp_embeddings = self.input_embedding(selected_index) # [1, inp_E]
            last_rnn_outp, hidden = self.rnn(inp_embeddings[None, :, :], hidden)
            last_rnn_outp = last_rnn_outp[0]

        return torch.tensor(generation, dtype=torch.long, device=device)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h = weight.new_zeros(self.n_layers, batch_size, self.n_hidden_units)
        c = weight.new_zeros(self.n_layers, batch_size, self.n_hidden_units)
        return (h, c)
