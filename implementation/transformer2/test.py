#!/usr/bin/env python3 
from config import global_config as cfg
import reader

def print_dialogues(r):
    """ Prints dialogues: user_input, bspan and machine response 
    for one batch. """

    for batch in r.mini_batch_iterator('train'):
        for d in batch: # first is turn0, then turn1

            print(reader.pad_sequences(d['user'], 128)) 
            return
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
def max_len(reader):
    x = 0
    sent = ''
    y = 0
    for batch in r.mini_batch_iterator('train'):
        for d in batch: # first is turn0, then turn1
            for user, u_len, m_len, degree in zip(d['user'], d['u_len'], d['m_len'], d['degree']):
                if x < u_len:
                    x = u_len
                    sent = user
                if y < m_len:
                    y = m_len
    print('u_len')
    print(x)
    print(r.vocab.sentence_decode(sent))
    print(y)

if __name__=='__main__':
    # cfg.init_handler('tsdf-kvret')
    # cfg.dataset = 'kvret'
    # r = reader.KvretReader()
    cfg.init_handler('tsdf-camrest')
    cfg.dataset = 'camrest'
    r = reader.CamRest676Reader()
    # max_len(r)
    print_dialogues(r)

