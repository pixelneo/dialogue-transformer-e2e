#!/usr/bin/env python3

import math
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import global_config as cfg
import reader


# TODO:
# 1. (maybe) do encoding for user and machine separately (additional positional encoding)
# 2. does torch transformer do teacher forcing? should it?
# 3. (solved) (HIGH PRIORITY) how to pass bspan to ResponseDecoder. Put it as an input and dont mask it. Make constant size for bspan (~ 20-30 words) and add some padding (new one?)
# 4. (A BIG TODO) probably a stupid question (ondra), but where is specified the size of input to transformer??? (either encoder, or decoder)
#       is it the `d_model` (=`ninp`) variable????
# 5. Sort out dimensions of inputs to en/decoders. This certainly is not working right now!! 
# 6. Do the training (define, loss, training loop, ...)

# Notes
# 1. EOS_Z1 ends section of bspan containing 'informables', EOS_Z2 ends 'requestables'
# 2. (much later) It would be possible to encode bspan (output from bspandecoder) and add it as another encoder, see: https://www.aclweb.org/anthology/W18-6326.pdf


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    """ User utterance encoder

    Args:
        ntoken: vocab size
        ninp: embedding dimension
        nhead: number of heads
        nhid: hidden layer size
        nlayers: number of layers
        dropout: dropout rate
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, params, dropout=0.5, embedding=None):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'TransformerEncoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding
        self.ninp = ninp
        self.params = params

        # self.init_weights()

    # def init_weights(self):
        # initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)

    def train(self, t=True):
        self.transformer_encoder.train(t)

    def forward(self, src):
        mask = src.eq(0).transpose(0,1)  # 0 corresponds to <pad>
        src = self.embedding(src) * self.ninp
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        return output

class BSpanDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, reader_, params, dropout=0.5, embedding=None):
        """
        Args:
            ntoken: vocab size
            ninp: embedding dimension
            nhead: number of heads
            nhid: hidden layer size
            nlayers: number of layers
            reader: instance of `Reader`
            dropout: dropout rate
        """
        super().__init__()
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'TransformerDecoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding
        self.ninp = ninp
        self.linear = nn.Linear(ninp, ntoken)
        self.reader_ = reader_
        self.params = params

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def train(self, t=True):
        self.transformer_decoder.train(t)

    def _generate_square_subsequent_mask(self, sz):
        """ This makes the model autoregressive.
        When decoding position t, look only at positions 0...t-1 """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory):
        """ Call decoder
        the decoder should be called repeatedly

        Args:
            tgt: input to transformer_decoder, shape: (seq, batch)
            memory: output from the encoder

        Returns:
            output from linear layer, (vocab size), pre softmax

        """
        tgt = tgt.long()
        go_tokens = torch.zeros((1, tgt.size(1)), dtype=tgt.dtype) + 3  # GO_2 token has index 3

        tgt = torch.cat([go_tokens, tgt], dim=0)  # concat GO_2 token along sequence lenght axis


        mask = tgt.eq(0).transpose(0,1)  # 0 corresponds to <pad>
        tgt = self.embedding(tgt) * self.ninp
        tgt = self.pos_encoder(tgt)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0))
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask)
        output = self.linear(output)
        return output

class ResponseDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, reader_, params, dropout=0.5, embedding=None):
        """
        Args:
            ntoken: vocab size
            ninp: embedding dimension
            nhead: number of heads
            nhid: hidden layer size
            nlayers: number of layers
            reader: instance of `Reader`
            dropout: dropout rate
        """
        super().__init__()
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'TransformerDecoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding
        self.ninp = ninp
        self.linear = nn.Linear(ninp, ntoken)
        self.reader_ = reader_
        self.params = params

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def train(self, t=True):
        self.transformer_decoder.train(t)

    def _generate_square_subsequent_mask(self, sz, bspan_size):
        # we do not mask the first positions (1 for degree, 1 for <go> token and 'some' for bspan)
        bspan_size = self.params['bspan_size']
        mask = (torch.triu(torch.ones(sz+1, sz+1), diagonal=-(bspan_size+1)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, bspan, degree):
        """ Call decoder

        Args:
            tgt: input to transformer_decoder, shape: (seq_len, batch)
            memory: output from the encoder
            degree: degree is the 'output from database', shape: (batch, cfg.degree_size)

        Returns:
            output from linear layer, (vocab size), pre softmax

        """

        go_tokens = torch.ones((1, tgt.size(1)), dtype=tgt.dtype)  # GO token has index 1
        degree_reshaped = torch.zeros((1, tgt.size(1), cfg.embedding_size), dtype=torch.float32)
        # print('tgt.shape0')
        # print(tgt.shape)
        # print('bspan.shape0')
        # print(bspan.shape)
        # print('degree_ershaped.shape0')
        # print(degree_reshaped.shape)
        # print('go_tokens.shape0')
        # print(go_tokens.shape)

        tgt = torch.cat([bspan, go_tokens, tgt], dim=0)  # concat bspan, GO and tokenstoken along sequence length axis
        # TODO pad `tgt` but also think of `degree` which is added later
        # print('tgt.shape')
        # print(tgt.shape)


        mask = torch.cat([torch.ones((1, tgt.size(1)), dtype=torch.int64), tgt]).eq(0).transpose(0,1)  # 0 corresponds to <pad>
        # TODO dimension are wrong
        # TODO also, final tgt dimension should be cfg.max_ts (128). however, now it is 128 before bspan is concatednated with it
        # mask = torch.cat([torch.ones((mask.size(0), 1)).bool(), mask])
        tgt = self.embedding(tgt) * self.ninp
        tgt = self.pos_encoder(tgt)
        # print('tgt.shape2')
        # print(tgt.shape)

        #    eg. [cheap restaurant EOS_Z1 EOS_Z2 PAD2 .... PAD2 01000 GO1 mask mask mask ..... ]
        #    ... [            ...          bspan    ... padding degree go     ....     masking ]


        # A BIG TODO: the size of `tgt` has to take the size of `bspan` (+1+1 for degree, go)  into account
        bspan_size = self.params['bspan_size']  # always the same
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), bspan_size)

        # tgt.size(1) is batch size (I know, why dim=1, but nn.Transformer wants it that way)
        degree_reshaped[0, :, :cfg.degree_size] = degree.transpose(0,1)  # add 1 more timestep (the first one as one-hot degree)
        tgt = torch.cat([degree_reshaped, tgt], dim=0)  # concat along sequence lenght axis
        # print('tgt.shape3')
        # print(tgt.shape)

        # THE ERROR: src_len is 150 and key_padding_mask.size(1) is 149
        # BOTH are wrong and should be 128 (currently max_len)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask)
        output = self.linear(output)
        # print('output.shape')
        # print(output.shape)
        return output


class SequicityModel(nn.Module):
    def __init__(self, encoder, bdecoder, rdecoder, params, reader_):
        super().__init__()
        self.model_type = 'Transformer'

        self.encoder = encoder
        self.bspan_decoder = bdecoder
        self.response_decoder = rdecoder

        self.reader_ = reader_
        self.params = params

    def train(self, t=True):
        super().train(t)
        self.encoder.train(t)
        self.bspan_decoder.train(t)
        self.response_decoder.train(t)

    def _greedy_decode_output(self, decoder, encoder_output, initial_decoder_input, eos_id, max_ts, response=False, bspan=None, degree=None):
        """ Autoregressive decoder: decode one step at a time, and run again with new word

        Args:
            decoder: instance of either bspan or response decoder
            encoder_output: output from encoder
            initial_decoder_input: target
            eos_id: id of either EOS_M, EOS_Z1, EOS_Z2
            max_ts: max timestep, different for BspanDec and ResponDec
            bspan: use only for ResponseDecoder
            degree: use only for ResponseDecoder

        Returns:
            a tensor, shape (seq_len, batch) with decoded output

        """
        input_ = initial_decoder_input  # shape (seq_len, batch)?
        pad_id = 0 if response else 4  # 4 is index for <pad2>
        decoded_sentences = torch.zeros_like(input_, dtype=torch.int64) * pad_id
        mask = torch.ones(input_.size(1)).bool()  # shape: batch
        for t in range(max_ts):
            if response:  # response decoder
                out = decoder(input_, encoder_output, bspan, degree)
            else:
                out = decoder(input_, encoder_output)

            # probs = nn.Softmax(out, dim=-1)  # may not be true: shape: (seq_len, batch, probs)

            probs = nn.functional.softmax(out[t,:,:], dim=-1)  # get prob for only t timestep (1,batch,probs)?
            _, inds = torch.topk(probs, 1)  # greedy decode (1, batch, 1)

            inds.squeeze_(-1)  # (1, batch, 1) -> (1, batch)
            inds.squeeze_(0)  # (batch)

            decoded_sentences[t,:].masked_scatter_(mask, inds)  # set decoded word at time step t for the whole batch

            input_[t, :] = inds

            # set mask if EOS is reached
            current_t_mask = (inds != eos_id)  # 0 if eos at t timestep
            mask = current_t_mask & mask

        return decoded_sentences



    def forward(self, user_input, bdecoder_input, rdecoder_input, degree):
        """ Call perform one step in sequicity.
        Encode input, decoder bspan, decode response 

        Args:
            user_input: input to encoder, should contain concatenated bspan (if exists)
            bdecoder_input: input to bspan decoder
            rdecoder_input: response
            degree: KB result

        Returns:

        """
        # transpose input to (seq_len, batch)
        # user_input = user_input.transpose(0,1)
        # bdecoder_input = bdecoder_input.transpose(0,1)
        # rdecoder_input = rdecoder_input.transpose(0,1)

        response_decoded = None
        encoded = self.encoder(user_input)

        if self.training:
            bdecoder_input = bdecoder_input
        else:
            bdecoder_input = torch.zeros(self.params['bspan_size'], bdecoder_input.size(1), dtype=torch.long) # go token is added later, in BSpanDecoder (seq_len, batch)

        # Even during training, we always have to decode BSpan, because we pass it to Response decoder
        bspan_decoded = self._greedy_decode_output(\
                                       self.bspan_decoder, \
                                       encoded, \
                                       bdecoder_input, \
                                       self.reader_.vocab.encode('EOS_Z2'),\
                                       self.params['bspan_size'])



        if self.training:
            # during training we will do only one pass through decoder and train on 
            # probabilities, outputs of softmax instead of one-hot decoded words.
            # TODO should we use decoded bspan or the supplied one? if supplied, we have to train BSpanDecoder somehow.
            response = self.response_decoder(rdecoder_input, encoded, bdecoder_input, degree)
        else:
            #response = self.response_decoder(concat, encoded, bspan_decoded, degree)
            response = self.response_decoder(rdecoder_input, encoded, bspan_decoded, degree)
            # response_decoded = self._greedy_decode_output(\
                                       # self.response_decoder, \
                                       # encoded, \
                                       # rdecoder_input, \
                                       # self.reader_.vocab.encode('EOS_M'),\
                                       # cfg.max_ts, \
                                       # True, \
                                       # bspan_decoded, \
                                       # degree)

        # TODO return only response or bspan also?
        return response, response_decoded



def remove_padding(tensor, dim=-1):
    """ Receives a tensor which is padded with zeros.
    Return a list of tensors where the trailing
    zeros have been removed.

    Args:
        tensor: tensor to remove padding on
        dim: on which dimension is the padding

    Example:
        > x = Tensor([
            [1 2 0 0]
            [3 0 1 0]
            [1 1 1 1]
          ])
        > remove_padding(x, dim=1)
        > list([ [1 2], [3], [1 1 1 1] ])

    """
    # TODO implement, ideally with torch.
    # this has low priority
    raise NotImplementedError()

def init_embedding_model(model, r):
    """ Set glove embeddings for model, r is a reader instance """
    initial_arr = model.embedding.weight.data.cpu().numpy()
    embedding_arr = torch.from_numpy(reader.get_glove_matrix(r.vocab, initial_arr))
    model.embedding.weight.data.copy_(embedding_arr)

def init_embedding(embedding, r):
    initial_arr = embedding.weight.data.cpu().numpy()
    embedding_arr = torch.from_numpy(reader.get_glove_matrix(r.vocab, initial_arr))
    embedding.weight.data.copy_(embedding_arr)
    return embedding

def convert_batch(batch, params):
    # convert batch to tensors
    # yield tensors with batched inputs
    # dict_keys(['dial_id', 'turn_num', 'user', 'response', 'bspan', 'u_len', 'm_len', 'degree', 'supervised'])

    for turn in batch:
        user = torch.zeros(params['user_size'],len(turn['user']), dtype=torch.long)
        bspan = torch.zeros(params['bspan_size'],len(turn['bspan']), dtype=torch.long)
        response = torch.zeros(params['response_size'], len(turn['response']), dtype=torch.long)
        degree = torch.zeros(5, len(turn['degree']), dtype=torch.long)
        for i, (u, b, r, d) in enumerate(zip(turn['user'], turn['bspan'], turn['response'], turn['degree'])):
            user[:len(u), i] = torch.tensor(u, dtype=torch.long)
            bspan[:len(b), i] = torch.tensor(b, dtype=torch.long)
            response[:len(r), i] = torch.tensor(r, dtype=torch.long)
            degree[:5,i] = torch.tensor(d, dtype=torch.long)

        yield user, bspan, response, degree


def get_params():
    # TODO: make parameter handling great again!
    # it would be better to use json or yaml (or something else) for setting parameters
    p = {}
    p['ntoken'] = cfg.vocab_size
    p['ninp'] = cfg.embedding_size
    p['nhead'] = 2
    p['nhid'] = 64
    p['nlayers'] = 3
    p['dropout_encoder'] = 0.2
    p['dropout_bdecoder'] = 0.2
    p['dropout_rdecoder'] = 0.2
    p['warm_lr'] = 0.1
    p['lr'] = 0.0001
    p['user_size'] = 128
    p['bspan_size'] = 20
    p['response_size'] = cfg.max_ts

    return p

def main_function(train_sequicity=True):
    cfg.init_handler('tsdf-camrest')
    cfg.dataset = 'camrest'
    r = reader.CamRest676Reader()
    params = get_params()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU or CPU

    embedding = nn.Embedding(params['ntoken'], params['ninp'])
    embedding = init_embedding(embedding, r)


    # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, reader, params, dropout=0.5, embedding=None):
    encoder = Encoder(
        params['ntoken'],\
        params['ninp'],\
        params['nhead'],\
        params['nhid'],\
        params['nlayers'],\
        params,\
        params['dropout_encoder'],\
        embedding).to(device)
    bspan_decoder = BSpanDecoder(
        params['ntoken'],\
        params['ninp'],\
        params['nhead'],\
        # params['nhid'] - params['bspan_size'] - 1,\
        params['bspan_size'],\
        params['nlayers'],\
        r,\
        params,\
        params['dropout_bdecoder'],\
        embedding).to(device)
    response_decoder = ResponseDecoder(
        params['ntoken'],\
        params['ninp'],\
        params['nhead'],\
        params['nhid'],\
        params['nlayers'],\
        r,\
        params,\
        params['dropout_rdecoder'],\
        embedding).to(device)

    model = SequicityModel(encoder, bspan_decoder, response_decoder, params, r)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    if train_sequicity:
        iterator = r.mini_batch_iterator('train') # bucketed by turn_num
        eval_iterator = r.mini_batch_iterator('dev')
        lowest_loss = 1000000000.0

        if not os.path.exists("models"):
            os.makedirs("models")
        for epoch in range(cfg.epoch_num):
            model.train()
            for batch in iterator:
                prev_bspan = None  # bspan from previous turn
                for user, bspan, response_, degree in convert_batch(batch, params):
                    optimizer.zero_grad()
                    out, _  = model(user, bspan, response_, degree)
                    # TODO what about OOV? like name_SLOT
                    r2 = torch.cat([response_, torch.zeros((22, out.size(1)), dtype=torch.int64)])
                    loss = criterion(out.view(-1, params['ntoken']), r2.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    print(loss)

            # TODO evaluate!!!
            model.eval()
            total_loss = 0.0
            softmax = torch.nn.Softmax(-1)
            for batch in iterator:
                prev_bspan = None  # bspan from previous turn
                for user, bspan, response_, degree in convert_batch(batch, params):
                    out, _ = model(user, bspan, response_, degree)
                    r2 = torch.cat([response_, torch.zeros((22, out.size(1)), dtype=torch.int64)])
                    loss = criterion(out.view(-1, 800), r2.view(-1))
                    # TODO it just does not work
                    decoded = torch.argmax(softmax(out), dim=-1)  # non autoregressive, just decode every timestep at once
                    for s in decoded:
                        x = r.vocab.sentence_decode(np.array(s))
                        print(x) # print sentences
                    total_loss += loss
                    print('Loss', loss)

            print("Total loss on test dataset:", total_loss)


            # TODO save model
            if total_loss < lowest_loss:
                lowest_loss = total_loss
                torch.save(model.state_dict(), "models/best_model.pt")

            model_path = "models/sequicity_epoch_" + str(epoch + 1) + ".pt"
            torch.save(model.state_dict(), model_path)

    else: # test the best model
        model.load_state_dict(torch.load("models/best_model.pt"))
        model.eval()
        iterator = r.mini_batch_iterator('test') 
        total_loss = 0.0
        softmax = torch.nn.Softmax(-1)
        for batch in iterator:
            prev_bspan = None  # bspan from previous turn

            for user, bspan, response_, degree in convert_batch(batch, params):
                out, _ = model(user, bspan, response_, degree)
                r2 = torch.cat([response_, torch.zeros((22, out.size(1)), dtype=torch.int64)])
                loss = criterion(out.view(-1, 800), r2.view(-1))
                # TODO it just does not work
                decoded = torch.argmax(softmax(out), dim=-1)  # non autoregressive, just decode every timestep at once
                for s in decoded:
                    x = r.vocab.sentence_decode(np.array(s))
                    print(x) # print sentences
                total_loss += loss
                print('Loss', loss)

        print("Total loss on test dataset:", total_loss)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    args = parser.parse_args()
    import random
    random.seed(1)
    torch.random.manual_seed(1)
    main_function(train_sequicity=args.train)
