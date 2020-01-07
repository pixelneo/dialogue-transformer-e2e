import torch

from torch import nn


# The official PyTorch Transformer implementation is missing positional
# encoding. See https://github.com/pytorch/pytorch/issues/24826. The following
# implementation of PositionalEncoding seems to be extremely popular.
# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_ter
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ------------------------------------------------------------------------------
# TODO: Do we need subsequent masking?
#
# Explanation of subsequent masking:
# https://nlp.seas.harvard.edu/2018/04/03/attention.html
#
# Below the attention mask shows the position each tgt word (row) is allowed to
# look at (column). Words are blocked for attending to future words during
# training.

# subsequent_mask implementation 1: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# subsequent_mask implementation 2: https://nlp.seas.harvard.edu/2018/04/03/attention.html
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
