#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    #TODO credit

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
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
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'TransformerEncoder'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * self.ninp
        src = self.pos_encoder(src)
        mask = src.eq(0)  # 0 corresponds to <pad>
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
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


def init_embedding(model, r):
    """ Set glove embeddings for model, r is a reader instance """
    initial_arr = model.embedding.weight.data.cpu().numpy()
    embedding_arr = torch.from_numpy(get_glove_matrix(r.vocab, initial_arr))
    model.embedding.weight.data.copy_(embedding_arr)

if __name__=='__main__':
    import numpy as np
    encoder = PositionalEncoding(10, 0.0)
    x = torch.ones([5,10])
    for i in range(1,5):
        x[i:,:] += torch.ones([5-i,10])
    print(x)
    print(encoder(x))



