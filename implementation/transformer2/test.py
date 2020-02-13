#!/usr/bin/env python3

# This file is only used for understanding reader.py

from config import global_config as cfg
import reader
import torch

def print_dialogues(r):
    """ Prints dialogues: user_input, bspan and machine response 
    for one batch. """

    for batch in r.mini_batch_iterator('test'):
        # print(len(batch))
        for d in batch: # first is turn0, then turn1
            # x = reader.get_glove_matrix
            print('s')
            print(d['turn_num'][0])
            print(len(d['user']))
            continue
            for dial_id, turn_num, user, bspan, response, u_len, m_len, degree in zip(d['dial_id'], d['turn_num'], \
                                                                                      d['user'], d['bspan'], d['response'], \
                                                                                      d['u_len'], d['m_len'], d['degree']):
                    print('dial_id: {}'.format(dial_id))
                    print('turn_num: {}'.format(turn_num))
                    x = r.vocab.sentence_decode(user)
                    print(x)
                    x = r.vocab.sentence_decode(bspan)
                    # print('bspan:', end=' ')
                    print(x)
                    x = r.vocab.sentence_decode(response)
                    print(x)
                    print('u_len: {}'.format(u_len))
                    print('m_len: {}'.format(m_len))
                    print('degree: {}'.format(degree))
                    print('------------')
            print('#'*50)

        print(d.keys())
        print('\n')
        print('*'*100)

def with_convert(r):
    for batch in r.mini_batch_iterator('train'):
        # print(len(batch))
        prev_z = None
        for d in batch: # first is turn0, then turn1
            u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
            m_len, degree_input, kw_ret \
                = reader._convert_batch(d, r, prev_z)
            # print(u_input)
            # print(d['user'])
            # return
            for ui, zi, mi, uio, zio, mio, deg in zip(u_input, z_input, m_input, d['user'], d['bspan'], d['response'], degree_input):
                # x = r.vocab.sentence_decode(ui)
                # print('u_input: {}'.format(x))
                x = r.vocab.sentence_decode(uio)
                print('u_origi: {}'.format(x))
                # x = r.vocab.sentence_decode(zi)
                # print('z_input: {}'.format(x))
                x = r.vocab.sentence_decode(zio)
                print('z_origi: {}'.format(x))
                # x = r.vocab.sentence_decode(mi)
                # print('m_input: {}'.format(x))
                x = r.vocab.sentence_decode(mio)
                print('m_origi: {}'.format(x))
                print('degree: {}'.format(deg))

                print('------------')
            print('#'*50)
            prev_z = d['bspan']

        # print(d.keys())
        print('\n')
        print('*'*100)

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

def test_subq(sz):
    mask = (torch.triu(torch.ones(sz+5, sz+5), -5) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def first_zero_index(tensor, dim=-1):
    mask = tensor == 0
    _, ind = torch.min(mask, dim=1)

if __name__=='__main__':
    # cfg.init_handler('tsdf-kvret')
    # cfg.dataset = 'kvret'
    # r = reader.KvretReader()
    cfg.init_handler('tsdf-camrest')
    cfg.dataset = 'camrest'
    r = reader.CamRest676Reader()
    # max_len(r)
    # # with_convert(r)
    # ten = torch.zeros(5, 6)
    # ten[:,:2] = torch.rand(5, 2)
    # ten[1,2] = 1.243
    # # ten[2,2] = 1.243
    # ten[2,3] = 1.243
    # print(ten)
    # # mask = ten == 0
    # # ten = ten[:,:].nonzero()
    # print(ten.split(2))

    print(test_subq(6))

