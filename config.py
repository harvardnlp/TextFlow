import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--run_name', type=str, default='charptb_AF-AF')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--load_dir', type=str)
    parser.add_argument('--evaluate_only', action='store_true')
    parser.add_argument('--dataset', type=str, default='ptb')
    parser.add_argument('--nll_every', type=int, default=5)
    parser.add_argument('--indep_bernoulli', action='store_true')
    parser.add_argument('--noT_condition', action='store_true')

    # Optimization parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--B_train', type=int, default=15)
    parser.add_argument('--B_val', type=int, default=15)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--seed', type=int)

    # KL/LR schedule parameters
    parser.add_argument('--initial_kl_zero', type=int, default=4)
    parser.add_argument('--kl_rampup_time', type=int, default=10)
    parser.add_argument('--patience', type=int, default=1)

    # Sample parameters
    parser.add_argument('--ELBO_samples', type=int, default=10)
    parser.add_argument('--nll_samples', type=int, default=30)

    # General model parameters
    parser.add_argument('--model_type', type=str, default='discrete_flow')
    parser.add_argument('--inp_embedding_size', type=int, default=500)
    parser.add_argument('--zsize', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dlocs', nargs='*', default=['prior_rnn'])
    parser.add_argument('--notie_weights', action='store_true')

    # Inference network parameters
    parser.add_argument('--q_rnn_layers', type=int, default=2)

    # Generative network parameters
    ## Prior
    parser.add_argument('--prior_type', type=str, default='AF')
    parser.add_argument('--p_ff_layers', type=int, default=0)
    parser.add_argument('--p_rnn_layers', type=int, default=2)
    parser.add_argument('--p_rnn_units', type=int, default=500)
    parser.add_argument('--p_num_flow_layers', type=int, default=1)
    parser.add_argument('--transform_function', type=str, default='nlsq')

    ### Prior MADE Flow
    parser.add_argument('--nohiddenflow', action='store_true')
    parser.add_argument('--hiddenflow_layers', type=int, default=2)
    parser.add_argument('--hiddenflow_units', type=int, default=100)
    parser.add_argument('--hiddenflow_flow_layers', type=int, default=5)
    parser.add_argument('--hiddenflow_scf_layers', action='store_true')

    ## Likelihood parameters
    parser.add_argument('--gen_bilstm_layers', type=int, default=2)

    args = parser.parse_args()

    if args.dlocs is None:
        setattr(args, 'dlocs', [])

    setattr(args, 'savedir', args.output_dir+'/'+args.run_name+'/saves/')
    setattr(args, 'logdir', args.output_dir+'/'+args.run_name+'/logs/')

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    if args.seed is None:
        setattr(args, 'seed', random.randint(0, 1000000))

    return args
