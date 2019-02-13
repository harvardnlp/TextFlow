import sys
import time
import random
import math
import json
import numpy as np

import torchtext
import torch
from torch.autograd import Variable

from baseline_model import LSTMModel
from discreteflow_model import DFModel
from config import parse_args
from utils import load_indep_bernoulli, load_categorical, get_optimizer, log, save, load, build_log_p_T, get_kl_weight


def run_epoch(train, start_kl_weight, delta_kl_weight, NLL_samples, ds, steps=-1):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    avg_kl = 0
    total_log_likelihood = 0
    total_tokens = 0
    start_time = time.time()

    accum_counter = 0
    for i, batch in enumerate(iter(ds)):
        if steps > 0 and i >= steps:
            break

        kl_weight = start_kl_weight + delta_kl_weight*i/len(ds)
        
        batch_data = Variable(batch.text[0].to(device))
        lengths = Variable(batch.text[1].to(device))

        if train and accum_counter == 0:
            model.zero_grad()

        if args.model_type == 'baseline':
            loss = model(batch_data, lengths)[:, :, None] # [T, B, s]
            kl_loss = torch.zeros_like(loss)
        elif args.model_type == 'discrete_flow':
            reconst_loss, kl_loss = model.evaluate_x(batch_data, lengths, ELBO_samples=args.ELBO_samples) # Inputs should be [T, B], outputs should be [T, B, s]
            loss = reconst_loss + kl_weight*kl_loss

        # Exact loss is -(ELBO(x_i)+log_p_T(T_i))/T_i for each x_i
        # NLL bound is 1/sum(T_i)*sum(-(ELBO(x_i)+log_p_T(T_i)))
        
        indices = torch.arange(batch_data.shape[0]).view(-1, 1).to(device)
        loss_mask = indices >= lengths.view(1, -1)
        loss_mask = loss_mask[:, :, None].repeat(1, 1, loss.shape[-1])
        
        loss[loss_mask] = 0
        kl_loss[loss_mask] = 0

        if not args.noT_condition:
            denom = (lengths+1).float() # if T conditioning, should normalizing by lengths+1 to be the same as normal <eos>-including models
        else:
            denom = (lengths).float()

        loss = loss.mean(-1).sum(0) # mean over ELBO samples and time, [B]
        if not args.noT_condition:
            loss -= log_p_T[lengths] # Take into account log_p_T for each batch (negative b/c this is NLL)

        obj = (loss/denom).mean() # Mean over batches

        if train:
            obj_per_accum = obj.clone()/args.grad_accum
            obj_per_accum.backward()

            accum_counter += 1
            if accum_counter == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                accum_counter = 0

        # Estimate NLL with importance sampling
        if NLL_samples > 0 and args.model_type == 'discrete_flow':
            with torch.no_grad():
                reconst_loss_val, kl_loss_val = model.evaluate_x(batch_data, lengths, ELBO_samples=NLL_samples) # [T, B, s]
                
                inside_terms = (-reconst_loss_val - kl_loss_val) # [T, B, s]
                loss_mask = indices >= lengths.view(1, -1)
                loss_mask = loss_mask[:, :, None].repeat(1, 1, NLL_samples)
                inside_terms[loss_mask] = 0

                inside_terms_sumT = inside_terms.sum(0) # [B, s]
                log_likelihood = torch.logsumexp(inside_terms_sumT, -1) - math.log(NLL_samples) # [B]

                if not args.noT_condition:
                    log_likelihood += log_p_T[lengths]
    
                total_log_likelihood += log_likelihood.sum().item()

        kl_loss = kl_loss.mean(-1).sum(0)
        total_loss += loss.sum().item()
        avg_kl += kl_loss.sum().item()
        total_tokens += denom.sum().item()

    avg_kl /= total_tokens
    total_loss /= total_tokens
    total_log_likelihood /= total_tokens

    return total_loss, avg_kl, total_log_likelihood, time.time()-start_time

# Setup
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Get parameters
device = torch.device('cuda')
args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# Load data
if args.indep_bernoulli:
    (train, val, test), pad_val, vocab_size = load_indep_bernoulli(args.dataset)
else:
    (train, val, test), pad_val, vocab_size = load_categorical(args.dataset, args.noT_condition)
log_p_T, max_T = build_log_p_T(args, train, val)
log_p_T = log_p_T.to(device)
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train, val, test), batch_sizes=[args.B_train, args.B_val, args.B_val], device=-1, repeat=False, sort_key=lambda x: len(x.text), sort_within_batch=True)

# Build model
loss_weights = torch.ones(vocab_size)
loss_weights[pad_val] = 0

if args.model_type == 'discrete_flow':
    prior_kwargs = {'p_rnn_layers': args.p_rnn_layers, 'p_rnn_units': args.p_rnn_units, 'p_num_flow_layers': args.p_num_flow_layers,
                    'nohiddenflow': args.nohiddenflow, 'hiddenflow_layers': args.hiddenflow_layers, 'hiddenflow_units': args.hiddenflow_units,
                    'hiddenflow_flow_layers': args.hiddenflow_flow_layers, 'hiddenflow_scf_layers': args.hiddenflow_scf_layers,
                    'transform_function': args.transform_function}
    model = DFModel(vocab_size, loss_weights, args.inp_embedding_size, args.hidden_size, args.zsize, args.dropout_p, args.dlocs,
                    args.prior_type, args.gen_bilstm_layers, prior_kwargs,
                    args.q_rnn_layers, not args.notie_weights, max_T, indep_bernoulli=args.indep_bernoulli).to(device)
elif args.model_type == 'baseline':
    model = LSTMModel(vocab_size, loss_weights, args.inp_embedding_size, args.hidden_size, args.p_rnn_layers, args.dropout_p, T_condition=not args.noT_condition,
                      max_T=max_T, tie_weights=not args.notie_weights, indep_bernoulli=args.indep_bernoulli).to(device)
    setattr(args, 'ELBO_samples', 1)
    setattr(args, 'nll_samples', 0)
    setattr(args, 'kl_rampup_time', 0)
    setattr(args, 'initial_kl_zero', 0)
else:
    raise ValueError('model_type must be one of discrete_flow, baseline')

# Build optimizer
optimizer = get_optimizer(args.optim, model.parameters(), args.lr)

# Load parameters if needed
if args.load_dir:
    starting_epoch, best_val_loss, lr = load(model, optimizer, args)
    auto_lr = True
    cur_impatience = 0
    optimizer = get_optimizer(args.optim, model.parameters(), lr)
else:
    starting_epoch = 0
    best_val_loss = 999999999
    lr = args.lr
    auto_lr = False

# If evaluate_only, only do that and don't train

if args.evaluate_only:
    torch.set_printoptions(threshold=10000)
    train_loss, train_kl, train_LL, train_time = run_epoch(False, 1.0, 0.0, args.nll_samples, train_iter, steps=200)
    print('train loss: %.5f, train NLL (%d): %.5f, train kl: %.5f, train time: %.2fs' % (train_loss, args.nll_samples, -train_LL, train_kl, train_time))

    val_loss, val_kl, val_LL, val_time = run_epoch(False, 1.0, 0.0, args.nll_samples, val_iter)
    print('val loss: %.5f, val NLL (%d): %.5f, val kl: %.5f, val time: %.2fs' % (val_loss, args.nll_samples, -val_LL, val_kl, val_time))

    test_loss, test_kl, test_LL, test_time = run_epoch(False, 1.0, 0.0, args.nll_samples, test_iter)
    print('test loss: %.5f, test NLL (%d): %.5f, test kl: %.5f, test time: %.2fs' % (test_loss, args.nll_samples, -test_LL, test_kl, test_time))

    sys.exit()

# Train
## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

log(args, '--------------------- NEW START ----------------------')

# Save parameters
with open(args.output_dir+'/'+args.run_name+'/params.txt', 'w') as f:
    f.write(json.dumps(args.__dict__, indent=4, sort_keys=True))


for i in range(starting_epoch, args.num_epochs):
    decrease_lr = False
    save_model = False
    
    last_kl_weight, _ = get_kl_weight(args, i-1)
    cur_kl_weight, done = get_kl_weight(args, i)

    if done:
        auto_lr = True
    
    train_loss, train_kl, _, train_time = run_epoch(True, last_kl_weight, cur_kl_weight-last_kl_weight, 0, train_iter)

    val_NLL_samples = args.nll_samples if (i+1)%args.nll_every == 0 else 0
    val_loss, val_kl, val_log_likelihood, val_time = run_epoch(False, cur_kl_weight, 0.0, val_NLL_samples, val_iter)
    
    log_str = 'Epoch %d | train loss: %.3f, val loss: %.3f, val NLL (%d): %.3f | train kl: %.3f, val kl: %.3f | kl_weight: %.3f, time: %.2fs/%.2fs' % \
        (i, train_loss, val_loss, val_NLL_samples, -val_log_likelihood, train_kl, val_kl, cur_kl_weight, train_time, val_time)
    print(log_str)
    log(args, log_str)

    if auto_lr:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cur_impatience = 0
            save_model = True
        else:
            cur_impatience += 1
            if cur_impatience == args.patience:
                decrease_lr = True
    
    if decrease_lr:
        lr /= 4
        optimizer = get_optimizer(args.optim, model.parameters(), lr)

        print('* Learning rate dropping by a factor of 4')
        log(args, '* Learning rate dropping by a factor of 4')
        cur_impatience = 0

    if save_model:
        save(model, optimizer, args, 'after_epoch_%d' % i, i+1, best_val_loss, lr)

save(model, optimizer, args, 'end', args.num_epochs, best_val_loss, lr)
