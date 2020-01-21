#!/usr/bin/env python3 
from config import global_config as cfg
import reader

def print_dialogues(reader):
    """ Prints dialogues: user_input, bspan and machine response 
    for one batch. """

    for batch in r.mini_batch_iterator('train'):
        for d in batch: # first is turn0, then turn1
            for dial_id, turn_num, user, bspan, response, u_len, m_len, degree in zip(d['dial_id'], d['turn_num'], \
                                                                                      d['user'], d['bspan'], d['response'], \
                                                                                      d['u_len'], d['m_len'], d['degree']):
                    print('dial_id: {}'.format(dial_id))
                    print('turn_num: {}'.format(turn_num))
                    x = r.vocab.sentence_decode(user)
                    print(x)
                    x = r.vocab.sentence_decode(bspan)
                    print(x)
                    x = r.vocab.sentence_decode(response)
                    print(x)
                    print('u_len: {}'.format(u_len))
                    print('m_len: {}'.format(m_len))
                    print('degree: {}'.format(degree))
                    print('------------')
        print(d.keys())
        return

if __name__=='__main__':
    cfg.init_handler('tsdf-camrest')
    cfg.dataset = 'camrest'
    r = reader.CamRest676Reader()
    print_dialogues(r)

