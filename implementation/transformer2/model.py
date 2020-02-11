#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import global_config as cfg


# TODO:
# 1. (maybe) do encoding for user and machine separately (additional positional encoding)


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

    def forward(self, src):
        src = self.embedding(src) * self.ninp
        src = self.pos_encoder(src)
        mask = src.eq(0)  # 0 corresponds to <pad>
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        return output

class BSpanDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, embedding=None):
        """
        Args:
            ntoken: vocab size
            ninp: embedding dimension
            nhead: number of heads
            nhid: hidden layer size
            nlayers: number of layers
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

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.transformer_decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        """ This makes the model autoregressive.
        When decoding position t, look only at positions 0...t-1 """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt):
        tgt = self.embedding(tgt) * self.ninp
        tgt = self.pos_encoder(tgt)
        mask = tgt.eq(0)  # 0 corresponds to <pad>
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0])
        output = self.transformer_decoder(tgt, tgt_mask=tgt_mask, src_key_padding_mask=mask)
        return output

class ResponseDecoder(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, embedding=None):
        """
        Args:
            ntoken: vocab size
            ninp: embedding dimension
            nhead: number of heads
            nhid: hidden layer size
            nlayers: number of layers
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

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.transformer_decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt):
        tgt = self.embedding(tgt) * self.ninp
        tgt = self.pos_encoder(tgt)
        mask = tgt.eq(0)  # 0 corresponds to <pad>
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0])
        output = self.transformer_decoder(tgt, tgt_mask=tgt_mask, src_key_padding_mask=mask) # TODO create tgt_mask, masking positions > timestep
        return output


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

    return p

def main_function():
    cfg.init_handler('tsdf-camrest')
    cfg.dataset = 'camrest'
    r = reader.CamRest676Reader()
    params = get_params()

    embedding = nn.Embedding(params['ntoken'], params['ninp'])
    embedding = init_embedding(embedding, r)

    encoder = Encoder(params['ntoken'], params['ninp'], params['nhead'], params['nhid'], params['dropout'], embedding)
    bspan_decoder = BSpanDecoder(params['ntoken'], params['ninp'], params['nhead'], params['nhid'], params['dropout'], embedding)
    response_decoder = BSpanDecoder(params['ntoken'], params['ninp'], params['nhead'], params['nhid'], params['dropout'], embedding)


if __name__=='__main__':
    main_function()
