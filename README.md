# Latent Normalizing Flows for Discrete Sequences

## Dependencies
The code was tested with `python 3.6`, `pytorch 0.4.1`, `torchtext 0.2.3`, and `CUDA 9.2`.

## Character-level language modeling:

PTB data with [Mikolov preprocessing](http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf) is checked in.

Baseline and proposed models can be trained with:
```
python main.py --dataset ptb --run_name charptb_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20
python main.py --dataset ptb --run_name charptb_discreteflow_af-af 
python main.py --dataset ptb --run_name charptb_discreteflow_af-scf --hiddenflow_scf_layers
python main.py --dataset ptb --run_name charptb_discreteflow_iaf-scf --dropout_p 0 --hiddenflow_flow_layers 3 --hiddenflow_scf_layers --prior_type IAF
```

Evaluation on the test set is run with e.g.
```
python main.py --dataset ptb --run_name charptb_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20 --load_dir output/charptb_baselinelstm/saves/<save> --evaluate
```

## Polyphonic music:

Commands to train the baseline and proposed models are

Nottingham:
```
python main.py --indep_bernoulli --dataset nottingham --run_name nottingham_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20 --patience 4
python main.py --indep_bernoulli --dataset nottingham --run_name nottingham_discreteflow_af-af --patience 2 --ELBO_samples 1 --B_train 5 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15
python main.py --indep_bernoulli --dataset nottingham --run_name nottingham_discreteflow_af-scf --patience 2 --ELBO_samples 1 --B_train 5 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --hiddenflow_scf_layers
python main.py --indep_bernoulli --dataset nottingham --run_name nottingham_discreteflow_iaf-scf --patience 2 --ELBO_samples 1 --B_train 5 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --hiddenflow_scf_layers --prior_type IAF
```

Piano_midi
```
python main.py --indep_bernoulli --dataset piano_midi --run_name piano_midi_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20 --patience 4
python main.py --indep_bernoulli --dataset piano_midi --run_name piano_midi_discreteflow_af-af --patience 2 --ELBO_samples 1 --B_train 1 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --nll_samples 4 --grad_accum 8
python main.py --indep_bernoulli --dataset piano_midi --run_name piano_midi_discreteflow_af-scf --patience 2 --ELBO_samples 1 --B_train 1 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --nll_samples 4 --grad_accum 8 --hiddenflow_scf_layers
python main.py --indep_bernoulli --dataset piano_midi --run_name piano_midi_discreteflow_iaf-scf --patience 2 --ELBO_samples 1 --B_train 1 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --nll_samples 4 --grad_accum 8 --hiddenflow_scf_layers --prior_type IAF
```

Musedata
```
python main.py --indep_bernoulli --dataset muse_data --run_name muse_data_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20 --patience 4
python main.py --indep_bernoulli --dataset muse_data --run_name muse_data_discreteflow_af-af --patience 2 --ELBO_samples 1 --B_train 1 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --nll_samples 4 --grad_accum 8
python main.py --indep_bernoulli --dataset muse_data --run_name muse_data_discreteflow_af-scf --patience 2 --ELBO_samples 1 --B_train 1 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --nll_samples 4 --grad_accum 8 --hiddenflow_scf_layers
python main.py --indep_bernoulli --dataset muse_data --run_name muse_data_discreteflow_iaf-scf --patience 2 --ELBO_samples 1 --B_train 1 --B_val 1 --initial_kl_zero 20 --kl_rampup_time 15 --nll_samples 4 --grad_accum 8 --hiddenflow_scf_layers --prior_type IAF
```

JSB_chorales
```
python main.py --indep_bernoulli --dataset jsb_chorales --run_name jsb_chorales_baselinelstm --model_type baseline --dropout_p 0.1 --optim sgd --lr 20 --patience 4
python main.py --indep_bernoulli --dataset jsb_chorales --run_name jsb_chorales_discreteflow_af-af --patience 2 --ELBO_samples 1 --B_train 5 --B_val 64 --initial_kl_zero 20 --kl_rampup_time 15
python main.py --indep_bernoulli --dataset jsb_chorales --run_name jsb_chorales_discreteflow_af-scf --patience 2 --ELBO_samples 1 --B_train 5 --B_val 64 --initial_kl_zero 20 --kl_rampup_time 15 --hiddenflow_scf_layers
python main.py --indep_bernoulli --dataset jsb_chorales --run_name jsb_chorales_discreteflow_iaf-scf --patience 2 --ELBO_samples 1 --B_train 5 --B_val 64 --initial_kl_zero 20 --kl_rampup_time 15 --hiddenflow_scf_layers --prior_type IAF
```
