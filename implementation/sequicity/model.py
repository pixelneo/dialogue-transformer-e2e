import torch
import random
import numpy as np
from config import global_config as cfg
from reader import CamRest676Reader, get_glove_matrix
from reader import KvretReader
from tsd_net import TSD, cuda_, nan
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from reader import pad_sequences
import argparse, time

from metric import CamRestEvaluator, KvretEvaluator
import logging


class Model:
    def __init__(self, dataset):
        reader_dict = {
            'camrest': CamRest676Reader,
            'kvret': KvretReader,
        }
        model_dict = {
            'TSD':TSD
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator,
            'kvret': KvretEvaluator,
        }
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                                   hidden_size=cfg.hidden_size,
                                   vocab_size=cfg.vocab_size,
                                   layer_num=cfg.layer_num,
                                   dropout_rate=cfg.dropout_rate,
                                   z_length=cfg.z_length,
                                   max_ts=cfg.max_ts,
                                   beam_search=cfg.beam_search,
                                   beam_size=cfg.beam_size,
                                   eos_token_idx=self.reader.vocab.encode('EOS_M'),
                                   vocab=self.reader.vocab,
                                   teacher_force=cfg.teacher_force,
                                   degree_size=cfg.degree_size,
                                   reader=self.reader)
        self.EV = evaluator_dict[dataset] # evaluator class
        if cfg.cuda: self.m = self.m.cuda()
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1

    def _convert_batch(self, py_batch, prev_z_py=None):
        u_input_py, u_len_py = py_batch['user'], py_batch['u_len']
        prev_z_input = None

        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size: prev_z_py[i][j] = 2 #unk

        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size: prev_z_py[i][j] = 2 #unk

            prev_z_input = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_input = {
                'np': prev_z_input,
                'len': np.array([len(k) for k in prev_z_py]),
                'tensor': cuda_(Variable(torch.from_numpy(prev_z_input).long()))
            }

        u_input = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        u_input = {
            'np': u_input,
            'len': np.array(u_len_py),
            'tensor': cuda_(Variable(torch.from_numpy(u_input).long()))
        }

        m_input = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose((1, 0))
        m_input = {
            'np': m_input,
            'len': np.array(py_batch['m_len']),
            'tensor': cuda_(Variable(torch.from_numpy(m_input).long()))
        }

        z_input = pad_sequences(py_batch['bspan'], padding='post').transpose((1, 0))
        z_input = {
            'np': z_input,
            'tensor': cuda_(Variable(torch.from_numpy(z_input).long()))
        }

        degree_input = cuda_(Variable(torch.from_numpy(np.array(py_batch['degree'])).float()))

        return u_input, z_input, m_input, prev_z_input, degree_input

    def train(self):
        lr, prev_min_loss, early_stop_count, train_time = cfg.lr, 1 << 30, cfg.early_stop_count, 0

        for epoch in range(cfg.epoch_num):
            start_time = time.time()
            if epoch <= self.base_epoch: continue

            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            sup_loss, sup_cnt, optim = 0, 0, self.optim

            for iter_num, dial_batch in enumerate(self.reader.mini_batch_iterator('train')):
                turn_states, prev_z = {}, None

                for turn_num, turn_batch in enumerate(dial_batch):
                    if cfg.truncated: logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()

                    u_input, z_input, m_input, prev_z_input, degree_input = self._convert_batch(turn_batch, prev_z)
                    loss, pr_loss, m_loss, turn_states = self.m(u_input, z_input, m_input, prev_z_input, turn_states, degree_input, mode='train')

                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    optim.step()

                    sup_loss += loss.item()
                    sup_cnt += 1
                    logging.debug(f'loss:{loss.item()} pr_loss:{pr_loss.item()} m_loss:{m_loss.item()} grad:{grad}')

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - start_time
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time()-start_time))
            valid_loss = valid_sup_loss + valid_unsup_loss
            self.save_model(epoch)
            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=5e-5)
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, z_input, m_input, prev_z_input, degree_input = self._convert_batch(turn_batch, prev_z)
                m_idx, z_idx, turn_states = self.m(u_input, z_input, m_input, prev_z_input, turn_states, degree_input, mode, dial_id=turn_batch['dial_id'])
                self.reader.wrap_result(turn_batch, m_idx, z_idx, prev_z=prev_z)
                prev_z = z_idx

        ev = self.EV(result_path=cfg.result_path)
        res = ev.run_metrics()
        self.m.train()
        return res

    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, z_input, m_input, prev_z_input, degree_input = self._convert_batch(turn_batch)
                loss, pr_loss, m_loss, turn_states = self.m(u_input, z_input, m_input, prev_z_input, turn_states, degree_input, mode='train')

                sup_loss += loss.item()
                sup_cnt += 1

                logging.debug(f'loss:{loss.item()} pr_loss:{pr_loss.item()} m_loss:{m_loss.item()}')

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        print('result preview...')
        self.eval()
        return sup_loss, unsup_loss

    def reinforce_tune(self):
        lr = cfg.lr
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()))
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        for epoch in range(self.base_epoch + cfg.rl_epoch_num + 1):
            mode = 'rl'
            if epoch <= self.base_epoch:
                continue
            epoch_loss, cnt = 0,0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim #Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=0)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    optim.zero_grad()

                    u_input, z_input, m_input, prev_z_input, degree_input = self._convert_batch(turn_batch, prev_z)
                    loss_rl = self.m(u_input, z_input, m_input, prev_z_input, turn_states, degree_input, mode, dial_id=turn_batch['dial_id'])

                    if loss_rl is not None:
                        loss = loss_rl #+ loss_mle * 0.1
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm(self.m.parameters(), 2.0)
                        optim.step()
                        epoch_loss += loss.data.cpu().numpy()[0]
                        cnt += 1
                        logging.debug(f'{mode} loss {loss.item()}, grad:{grad}')

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = epoch_loss / (cnt + 1e-8)
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            valid_loss = valid_sup_loss + valid_unsup_loss

            #self.save_model(epoch)

            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def save_model(self, epoch, path=None, critical=False):
        if not path:
            path = cfg.model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cpu')
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.z_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.m_decoder.emb.weight.data.copy_(embedding_arr)

    def count_params(self):

        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])

        print('total trainable params: %d' % param_cnt)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-model')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.init_handler(args.model)
    cfg.dataset = args.model.split('-')[-1]

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.info(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.info('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(args.model.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
    elif args.mode == 'test':
        m.load_model()
        m.eval()
    elif args.mode == 'rl':
        m.load_model()
        m.reinforce_tune()


if __name__ == '__main__':
    main()
