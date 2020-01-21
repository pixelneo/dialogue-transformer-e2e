#!/usr/bin/env python3 
from config import global_config as cfg
import reader

cfg.init_handler('tsdf-camrest')
cfg.dataset = 'camrest'
r = reader.CamRest676Reader()

for batch in r.mini_batch_iterator('train'):
    for d in batch:
        for user, bspan, response in zip(d['user'], d['bspan'], d['response']):
            x = r.vocab.sentence_decode(user)
            print(x)
            x = r.vocab.sentence_decode(bspan)
            print(x)
            x = r.vocab.sentence_decode(response)
            print(x)
            print('------------')
        print(d.keys())
    exit()
