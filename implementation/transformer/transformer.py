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


class SimpleTransformerEncoder(nn.Module):
    """
    Standard implementation of a Transformer Encoder.
    Follows:
        - Original paper diagram: https://arxiv.org/pdf/1706.03762.pdf, Figure 1
        - Tutorials:
            - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
            - https://andrewpeng.dev/transformer-pytorch/
            - https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, vocabulary_size, dropout):
        super().__init__()

        # TODO: Do we want to use the dropout functionality of
        # PositionalEncoder, TransformerEncoderLayer, or both?

        D_MODEL = 512
        # d_model: The following citations from
        # https://arxiv.org/pdf/1706.03762.pdf explains what is d_model:
        #
        # The encoder is composed of a stack of N = 6 identical
        # layers. Each layer has two sub-layers. The first is a multi-head
        # self-attention mechanism, and the second is a simple, positionwise
        # fully connected feed-forward network.
        #
        # The decoder is also composed of a stack of N = 6 identical layers. In
        # addition to the two sub-layers in each encoder layer, the decoder
        # inserts a third sub-layer, which performs multi-head attention over
        # the output of the encoder stack
        #
        # All sub-layers in the model, as well as the embedding
        # layers, produce outputs of dimension d_model = 512.

        self.model_type='TransformerEncoder'

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_encoder = PositionalEncoder(
            d_model=D_MODEL,
            dropout,
            max_len=self.cfg.max_ts
        )

        self.transformer_encoder = nn.modules.TransformerEncoder(
            nn.modules.TransformerEncoderLayer(
                d_model=D_MODEL,  # 512 used in the original paper.
                nhead=8,  # Multi-head attention head count. 8 used in the original paper.
                dropout=dropout,
            ),
            num_layers=6  # Number of sub-encoder-layers in the encoder. 6 used in the original paper.
        )

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        :param input_seqs: Variable of [T,B]
            = a batch of input sequences.
            = 2D numpy array of dimension = (Max input sequence length x batch size.)
        :param input_lens: *numpy array* of len for each input sequence
        """
        # TODO: Is input_seqs already padded? I guess must be, since it is
        # numpy and thus a square matrix. What is the padding token?

        embedded = self.embedding(input_seqs)
        embedded = self.positional_encoder(embedded)

        # TODO: We need to use `input_lens`, create some masks and pass them to
        # `transformer_encoder`:
        #   - mask: the mask for the src sequence (optional).
        #   - src_key_padding_mask: the mask for the src keys per batch (optional).
        #
        # src_key_padding_mask: For the example, this looks like [False, False,
        # False, False, False, False, False, True, True, True] where the True
        # positions should be masked. That is input_len * [False] +
        # (max_input_len - input_len) * [True]?
        outputs = self.transformer_encoder(embedded)

        return outputs, embedded


class SimpleTransformerDecoder(nn.Module):
    """
    We are now replacing both BSpanDecoder and ResponseDecoder by this SimpleTransformerDecoder.
    Although BSpanDecoder and ResponseDecoder are similar, BSpanDecoder has one
    extra feature: It uses the previous encoder endput as an extra input.
    TODO: As future work, could we implement the same trick in our Transformer version?
    """
    def __init__(self, vocabulary_size, dropout):
        super().__init__()

        self.model_type='TransformerDecoder'

        self.positional_encoder = PositionalEncoder(d_model, dropout, self.cfg.max_ts)
        decoder_layers = nn.modules.TransformerDecoderLayer(d_model, nhead, dim_ff)
        self.transformer_decoder = nn.modules.TransformerDecoder(decoder_layers, n_layers)

        self.dropout_rate = dropout_rate
        self.inp_dropout = nn.Dropout(self.dropout_rate)

        self.vocab = vocab


    # TODO: THIS function IS a complete WORK-IN-PROGRESS
    def forward(self, z_enc_out, u_enc_out, u_input_np, m_t_input, degree_input, last_hidden, z_input_np):
        sparse_z_input = Variable(self.get_sparse_selective_input(z_input_np), requires_grad=False)

        m_embed = self.emb(m_t_input)

        # z_enc_out are stacked (along the time-step axis) belief spans (computed in BSpanDecoder)
        # TODO: Why should this be a BSpanDecoder output, when the variable states:
        #   - z_ = belief span
        #   - enc_out = encoder output
        #   - Would a better name be z_dec_out, or is this incorrect understanding of the code?
        z_context = self.attn_z(last_hidden, z_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_Z2')],
                                inp_seqs=z_input_np)

        # u_enc_out are "user encoder output" = user inputs encoded by the Encoder (SimpleDynamicEncoder)
        u_context = self.attn_u(last_hidden, u_enc_out, mask=True, stop_tok=[self.vocab.encode('EOS_M')],
                                inp_seqs=u_input_np)

        # TODO what is degree_input? it is coming from the data
        transformer_in = torch.cat([m_embed, u_context, z_context, degree_input.unsqueeze(0)], dim=2)
        # TODO: How exactly does last_hidden work?
        gru_out, last_hidden = self.gru(transformer_in, last_hidden)


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
