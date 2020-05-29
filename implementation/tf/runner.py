#!/usr/bin/env python3 
import itertools

import neptune
import yaml

from transformer import *
from reader import *

def load_params(path):
    """ Get input for parameter search and return grid-search inputs

    Returns:
        List with parameter names and iterator over all combinations of parameters.

    """
    def _params(p_names, it):
        for p_set in it:
             yield dict(zip(p_names, p_set))

    with open(path, 'r') as f:
        obj = yaml.safe_load(f)
    it = itertools.product(*[obj[p] for p in obj['parameters']])
    return obj['parameters'], _params(obj['parameters'], it)

def use_param(p):
    """ Should this set of parameters be used (has it not been used before)? """
    with open('log/used_params.txt') as f:
        used_params = dict([(l.strip(), None) for l in f.readlines()])
    if str(p) not in used_params:
        return True
    return False

def log_param(p):
    """ Log this set of parameters as used """
    with open('log/used_params.txt', 'a') as f:
        print(p, file=f)


if __name__ == "__main__":
    # def __init__(self, vocab_size, num_layers=3, d_model=50, dff=512, num_heads=5, dropout_rate=0.1, reader=None):
    parameters, it = load_params('params.yaml')

    neptune.init('dialogue-transformer-e2e/runs')

    ds = "tsdf-camrest"
    cfg.init_handler(ds)
    cfg.dataset = ds.split('-')[-1]
    reader = CamRest676Reader()

    for params in it:  # iterate over all possible parameter combinations
        if not use_param(params):
            continue
        neptune.create_experiment(name='parameter_search', params=params)
        print(params)

        model = SeqModel(vocab_size=cfg.vocab_size, reader=reader, num_layers=params['num_layers'], dff=params['dim_ff'], num_heads=params['num_heads'] )
        model.train_model(log=True)
        log_param(params)

