#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import global_config as cfg


# TODO:
# 1. (maybe) do encoding for user and machine separately (additional positional encoding)
# 2. does torch transformer do teacher forcing? should it?

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
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, embedding=None):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'TransformerEncoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding
        self.ninp = ninp

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.transformer_encoder.weight.data.uniform_(-initrange, initrange)

    def train(self, t):
        self.transformer_encoder.train(t)

    def forward(self, src):
        src = self.embedding(src) * self.ninp
        src = self.pos_encoder(src)
        mask = src.eq(0)  # 0 corresponds to <pad>
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        return output

class BSpanDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, reader, dropout=0.5, embedding=None):
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
        self.transformer_decoder = TransformerDecoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding
        self.ninp = ninp
        self.linear = nn.Linear(ninp, ntoken)
        self.reader = reader

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.transformer_decoder.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def train(self, t):
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
        go_tokens = torch.zeros((1, tgt.size(1)) + 3  # GO_2 token has index 3
        tgt = torch.cat([go_tokens, tgt], dim=0)  # concat GO_2 token along sequence lenght axis

        tgt = self.embedding(tgt) * self.ninp
        tgt = self.pos_encoder(tgt)
        mask = tgt.eq(0)  # 0 corresponds to <pad>
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0))
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask)
        output = self.linear(output)
        return output

class ResponseDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, reader, dropout=0.5, embedding=None):
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
        self.transformer_decoder = TransformerDecoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding
        self.ninp = ninp
        self.linear = nn.Linear(ninp, ntoken)
        self.reader = reader

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.transformer_decoder.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def train(self, t):
        self.transformer_decoder.train(t)

    def _generate_square_subsequent_mask(self, sz, bspan_size):
        # we do not mask first 2 positions (1 for degree, 1 for <go> token)
        mask = (torch.triu(torch.ones(sz+1+bspan_size, sz+1+bspan_size), diagonal=-(bspan_size+1)) == 1).transpose(0, 1)
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

        go_tokens = torch.ones((1, tgt.size(1))  # GO token has index 1
        
        tgt = torch.cat([bspan, go_tokens, tgt], dim=0)  # concat bspan, GO and tokenstoken along sequence lenght axis

        tgt = self.embedding(tgt) * self.ninp
        tgt = self.pos_encoder(tgt)
        mask = tgt.eq(0)  # 0 corresponds to <pad>
        
        # TODO Bspan Size (can be different for every turn in batch?)
        # TODO how to solve this, when there is one padding mask for a batch?
        #    we do not mask bspan, degree and go token.
        #    eg. [cheap restaurant EOS_Z1 EOS_Z2 01000 GO1 mask mask mask ..... ]
        
        bspan_size = None
        raise NotImplementedError()
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(0), bspan_size)


        # tgt.size(1) is batch size (I know, why dim=1, but nn.Transformer wants it that way)
        degree_reshaped = torch.zeros(1, tgt.size(1), cfg.embedding_size)
        degree_reshaped[:,:, :cfg.degree_size] = degree  # add 1 more timestep (the first one as one-hot degree)
        tgt = torch.cat([degree_reshaped, tgt], dim=0)  # concat along sequence lenght axis

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=mask)
        output = self.linear(output)
        return output

def SequicityModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, reader, dropout=0.5):
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
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(ntoken, ninp) if embedding is None else embedding

        self.encoder = Encoder(ntoken, ninp, nhead, nhid, dropout)
        self.bspan_decoder = BSpanDecoder(ntoken, ninp, nhead, nhid, reader, dropout, embedding)
        self.response_decoder = BSpanDecoder(ntoken, ninp, nhead, nhid, reader, dropout, embedding)

        self.reader = reader

    def train(self, t):
        super().train(t)
        self.encoder.train(t)
        self.bspan_decoder.train(t)
        self.response_decoder.train(t)

    def _greedy_decode_output(self, decoder, encoder_output, initial_decoder_input, eos_id):
        """ Iteratively run decoder until `eos_id` is reached

        Args:
            decoder: instance of either bspan or response decoder
            encoder_output: output from encoder
            initial_decoder_input: either <go>/<go2> symbol or "degree"<go> etc.
            eos_id: either id of EOS_M, EOS_Z1, EOS_Z2

        """
        for t in range(cfg.max_ts):
            if current_word == eos_id:
                break
            # TODO finish
        raise NotImplementedError()

    def forward(self, user_input, bdecoder_input, rdecoder_input, degree):
        """ Call perform one step in sequicity.
        Encode input, decoder bspan, decode response 

        Args:

        Returns:

        """
        # transpose input to (seq_len, batch)
        user_input = user_input.transpose(0,1)
        bdecoder_input = bdecoder_input.transpose(0,1)
        rdecoder_input = rdecoder_input.transpose(0,1)

        encoded = self.encoder(user_input)

        if self.training
            # TODO decode first EOS_Z1 and then EOS_Z2 = seq_len steps
            bdecoder_input = bdecoder_input
            raise NotImplementedError()
        else:
            bdecoder_input = torch.zeros(cfg.max_ts, bdecoder_input.size(1)) #(seq_len, batch)
            # bdecoder_input[:,  # finish (ondra)
            raise NotImplementedError()
        bspan = self.bspan_decoder(bdecoder_input, encoded)

        # TODO move this to _greedy_decode_output
        bspan_decoded = nn.Softmax(bspan, dim=-1).transpose(0,1)  # (batch_size, seq_len) 
        bspan_decoded_no_padding = remove_padding(bspan_decoded, dim=1)

        if self.training:
            # during training we will do only one pass through decoder and train on 
            # probabilities, outputs of softmax instead of one-hot decoded words.
            #response = self.response_decoder(concat, encoded, bdecoder_input, degree)
        else:
            # use decoded bspan instead of the supplied one
            # TODO concat `decoded_bspan` with degree and r_decoder_input
            #response = self.response_decoder(concat, encoded, bspan_decoded, degree)
            raise NotImplementedError()

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
    # this has lower priority
    raise NotImplementedError()

def init_embedding_model(model, r):
    """ Set glove embeddings for model, r is a reader instance """
    initial_arr = model.embedding.weight.data.cpu().numpy()
    embedding_arr = torch.from_numpy(get_glove_matrix(r.vocab, initial_arr))
    model.embedding.weight.data.copy_(embedding_arr)

def init_embedding(embedding, r):
    initial_arr = embedding.weight.data.cpu().numpy()
    embedding_arr = torch.from_numpy(get_glove_matrix(r.vocab, initial_arr))
    embedding.weight.data.copy_(embedding_arr)
    return embedding

def get_params():
    p = {}
    p['ntoken'] = cfg.vocab_size
    p['ninp'] = cfg.embedding_size
    p['nhead'] = 4
    p['nhid'] = 64
    p['nlayers'] = 3
    p['dropout'] = 0.2
    p['warm_lr'] = 0.1
    p['lr'] = 0.0001

    return p

def main_function():
    cfg.init_handler('tsdf-camrest')
    cfg.dataset = 'camrest'
    r = reader.CamRest676Reader()
    params = get_params()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU or CPU

    embedding = nn.Embedding(params['ntoken'], params['ninp'])
    embedding = init_embedding(embedding, r)

    encoder = Encoder(params['ntoken'], params['ninp'], params['nhead'], params['nhid'], params['dropout'], embedding).to(device)
    bspan_decoder = BSpanDecoder(params['ntoken'], params['ninp'], params['nhead'], params['nhid'], params['dropout'], embedding).to(device)
    response_decoder = BSpanDecoder(params['ntoken'], params['ninp'], params['nhead'], params['nhid'], params['dropout'], embedding).to(device)

    optimizer = torch.optim.Adam([encoder, lr=params['lr'])

    iterator = r.mini_batch_iterator('train') # bucketed by turn_num
    # TODO what about different batch sizes
    for batch in iterator:
        prev_bspan = None  # bspan from previous turn
        for turn in batch:
            encoder_input, encoder_input_np, bdec_input, rdec_input, rdec_input_np, encoder_len, \
            response_len, degree_input, kw_ret = reader._convert_batch(d, r, prev_bspan)

            # TODO implement training

            prev_bspan = turn['bspan']



if __name__=='__main__':
    main_function()
