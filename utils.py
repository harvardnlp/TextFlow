import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function

import torchtext
from torchtext import data
import io
import os

import numpy as np

from data.indep_bernoulli.load_data import load_data

# Data processing
# ------------------------------------------------------------------------------------------------------------------------------

class SentenceLanguageModelingDataset(data.Dataset):
    def __init__(self, path, text_field, encoding='utf-8', include_eos=True, **kwargs):
        fields = [('text', text_field)]
        examples = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                text = text_field.preprocess(line)
                if include_eos:
                    text += [u'<eos>']
                examples.append(data.Example.fromlist([text], fields))

        super().__init__(examples, fields, **kwargs)

def load_indep_bernoulli(dataset):
    dset = load_data(dataset)

    class MultipleOutputExample:
        def __init__(self, tensor):
            self.text = tensor

    class MultipleOutputField(data.Field):
        def __init__(self, pad_index):
            super().__init__(include_lengths=True, use_vocab=False)
            self.pad_index = pad_index

        def process(self, batch, device, train):
            lengths = [len(batch_i) for batch_i in batch]
            max_length = max(lengths)

            D = batch[0].shape[1]

            new_list = []
            for seq in batch:
                if len(seq) < max_length:
                    padding = torch.zeros(1, D)
                    padding[0, self.pad_index] = 1.
                    padding = padding.repeat(max_length-len(seq), 1)
                    seq = torch.cat((seq, padding), 0)
                new_list.append(seq)

            tensor = torch.stack(new_list)
            tensor = torch.transpose(tensor, 0, 1)

            lengths = torch.tensor(lengths)

            return tensor, lengths

    pad_val = 0
    text = MultipleOutputField(pad_val)

    datasets = {}
    for split, split_data in dset.items():
        examples = []
        for seq in split_data['sequences']:
            new_seq = torch.cat((torch.zeros(seq.shape[0], 1), seq), dim=1)
            examples.append(MultipleOutputExample(new_seq))
        datasets[split] = data.Dataset(examples, [('text', text)])

    train = datasets['train']
    val = datasets['valid']
    test = datasets['test']

    vocab_size = 89

    return (train, val, test), pad_val, vocab_size
        
def load_categorical(dataset, noT_condition_prior):
    unk_token = '<unk>'
    text = torchtext.data.Field(include_lengths=True, unk_token=unk_token, tokenize=(lambda s: list(s.strip())))


    MAX_LEN = 288
    MIN_LEN = 1

    train, val, test = SentenceLanguageModelingDataset.splits(path='./data/%s/'%dataset, train='train.txt', validation='valid.txt', test='test.txt', text_field=text,
                                                              include_eos=noT_condition_prior, filter_pred=lambda x: len(vars(x)['text']) <= MAX_LEN and len(vars(x)['text']) >= MIN_LEN)


    text.build_vocab(train)
    pad_val = text.vocab.stoi['<pad>']

    vocab_size = len(text.vocab)

    return (train, val, test), pad_val, vocab_size

# Utility functions
# ------------------------------------------------------------------------------------------------------------------------------

def get_optimizer(name, parameters, lr):
    if name == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, lr=lr)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr)
    else:
        raise NotImplementedError('Only adadelta, adam, and sgd, have been implemented')

    return optimizer

def log(args, log_str):
    with open(args.logdir+'summary.txt', 'a+') as f:
        f.write(log_str+'\n')

def save(model, optimizer, args, name, current_epoch, best_val, lr):
    savedir = args.savedir+name
    os.makedirs(savedir, exist_ok=True)

    torch.save(model.state_dict(), savedir+'/model.pt')
    torch.save(optimizer.state_dict(), savedir+'/optimizer.pt')
    np.savez(savedir+'/misc.npz', current_epoch=current_epoch, best_val=best_val, current_lr=lr)

def load(model, optimizer, args):
    model.load_state_dict(torch.load(args.load_dir+'/model.pt'))

    try:
        optimizer.load_state_dict(torch.load(args.load_dir+'/optimizer.pt'))
        misc_data = np.load(args.load_dir+'/misc.npz')
        current_epoch = misc_data['current_epoch']
        best_val = misc_data['best_val']
        current_lr = misc_data['current_lr']
    except:
        print('Error loading optimizer state. Will continue anyway starting from beginning')
        current_epoch, best_val, current_lr = 0, 999999999, args.lr

    return current_epoch, best_val, current_lr

def build_log_p_T(args, train, val):
    T_hist = torch.zeros(100000)
    max_T = 0
    for ex in train.examples+val.examples:
        ex_len = len(ex.text)
        T_hist[ex_len] += 1
        if ex_len > max_T:
            max_T = ex_len

    if args.indep_bernoulli:
        max_T = int(max_T*1.25)
        T_hist += 1

    T_hist = T_hist[:max_T+1]
    log_p_T = torch.log(T_hist/T_hist.sum())

    return log_p_T, max_T

def get_kl_weight(args, i):
    if args.initial_kl_zero == 0 and args.kl_rampup_time == 0:
        return 1.0, True

    x_start = args.initial_kl_zero
    x_end = args.initial_kl_zero + args.kl_rampup_time
    y_start = 0.00001
    y_end = 1.0
    done = False
    if i < x_start:
        cur_kl_weight = y_start
    elif i > x_end:
        cur_kl_weight = y_end
        done = True
    else:
        cur_kl_weight = (i-x_start)/(x_end-x_start)*(y_end-y_start) + y_start

    return cur_kl_weight, done

# Model utility functions
# ------------------------------------------------------------------------------------------------------------------------------

def make_pos_cond(T, B, lengths, max_T):
    device = lengths.device

    p_plus_int = torch.arange(T, device=device)[:, None].repeat(1, B)[:, :, None]
    p_plus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_plus_oh.scatter_(2, p_plus_int, 1)
    
    p_minus_int = lengths[None, :] - 1 - torch.arange(T, device=device)[:, None]
    p_minus_int[p_minus_int < 0] = max_T-1
    p_minus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_minus_oh.scatter_(2, p_minus_int[:, :, None], 1)
    
    pos_cond = torch.cat((p_plus_oh, p_minus_oh), -1) # [T, B, max_T*2]

    return pos_cond

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    if batch_first:
        inputs = inputs.transpose(0, 1)

    if inputs.size(1) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')

    reversed_inputs = inputs.data.clone()
    for i, length in enumerate(lengths):
        time_ind = torch.LongTensor(list(reversed(range(length))))
        reversed_inputs[:length, i] = inputs[:, i][time_ind]

    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
        
    return reversed_inputs

