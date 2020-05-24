import tensorflow as tf
import time
import numpy as np
from reader import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


np.set_printoptions(suppress=True)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', input_shape=(None, d_model)),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def read_embeddings(reader, embeddings_file="data/glove.6B.{}d.txt", embedding_size=50):
    """
    :param reader: a dialogue dataset reader, where we will get words mapped to indices
    :param embeddings_file: file path for glove embeddings
    :return: dictionary of indices mapped to their glove embeddings
    """
    vocab_to_index = {reader.vocab.decode(id): id for id in range(cfg.vocab_size)}
    embedding_matrix = np.zeros((cfg.vocab_size, embedding_size))
    embeddings_file = embeddings_file.format(embedding_size)
    with open(embeddings_file) as infile:
        for line in infile:
            word, coeffs = line.split(maxsplit=1)
            if word in vocab_to_index:
                word_index = vocab_to_index[word]
                embedding_matrix[word_index] = np.fromstring(coeffs, 'f', sep=' ')

    return embedding_matrix


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, embeddings_matrix=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if embeddings_matrix:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,
                                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix))
        else:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, embeddings_matrix=None):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if embeddings_matrix:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model,
                                                       embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix))
        else:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, embeddings_matrix=None):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate, embeddings_matrix)

        self.response_decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate, embeddings_matrix)

        self.bspan_decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate, embeddings_matrix)

        self.response_final = tf.keras.layers.Dense(target_vocab_size)
        self.bspan_final = tf.keras.layers.Dense(target_vocab_size)

    def bspan(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.bspan_decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        bspan_output = self.response_final(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return bspan_output, attention_weights

    def response(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.response_decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        response_output = self.response_final(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return response_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


class SeqModel:
    def __init__(self, vocab_size, num_layers=4, d_model=50, dff=512, num_heads=5, dropout_rate=0.1, reader=None):
        self.vocab_size = vocab_size
        input_vocab_size = vocab_size
        target_vocab_size = vocab_size

        self.learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.bspan_loss = tf.keras.metrics.Mean(name='train_loss')
        self.response_loss = tf.keras.metrics.Mean(name='train_loss')
        self.bspan_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.response_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        if reader:
            print("Reading pre-trained word embeddings with {} dimensions".format(d_model))
            embeddings_matrix = read_embeddings(reader, embedding_size=d_model)
        else:
            print("Initializing without pre-trained embeddings.")
            embeddings_matrix=None

        self.transformer = Transformer(num_layers, d_model, num_heads, dff,
                                  input_vocab_size, target_vocab_size,
                                  pe_input=input_vocab_size,
                                  pe_target=target_vocab_size,
                                  rate=dropout_rate, embeddings_matrix=embeddings_matrix)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def train_step_bspan(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer.bspan(inp=inp, tar=tar_inp, training=True,
                                                    enc_padding_mask=enc_padding_mask, look_ahead_mask=combined_mask,
                                                    dec_padding_mask=dec_padding_mask)

            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        gradients =[grad if grad is not None else tf.zeros_like(var)
                    for grad, var in zip(gradients, self.transformer.trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.bspan_loss(loss)
        self.bspan_accuracy(tar_real, predictions)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def train_step_response(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer.response(inp=inp, tar=tar_inp, training=True,
                                                       enc_padding_mask=enc_padding_mask, look_ahead_mask=combined_mask,
                                                       dec_padding_mask=dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        gradients =[grad if grad is not None else tf.zeros_like(var)
                    for grad, var in zip(gradients, self.transformer.trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.response_loss(loss)
        self.response_accuracy(tar_real, predictions)

    def evaluate(self, inp_sentence, MAX_LENGTH=40):
        start_token = [self.vocab_size]
        end_token = [self.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer.encode_as_ids(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.tokenizer.GetPieceSize() + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def restore_latest(self, checkpoint_path="./checkpoints/train"):
        ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')


def tensorize(id_lists):
    tensorized = tf.ragged.constant([x for x in id_lists]).to_tensor()
    return tf.cast(tensorized, dtype=tf.int32)


# TODO change these functions so that they can take tensor input and not just list
def produce_bspan_decoder_input(previous_bspan, previous_response, user_input):
    inputs =[]
    for counter, (x, y, z) in enumerate(zip(previous_bspan, previous_response, user_input)):
        new_sample = x + y + z  # TODO concatenation should be more readable than this
        inputs.append(new_sample)
    return tensorize(inputs)


def produce_response_decoder_input(previous_bspan, previous_response, user_input, bspan, kb):
    inputs = [a + b + c + d + e for a, b, c, d, e in zip(previous_bspan, previous_response, user_input, bspan, kb)]
    return tensorize(inputs)


if __name__ == "__main__":
    ds = "tsdf-camrest"
    cfg.init_handler(ds)
    cfg.dataset = ds.split('-')[-1]
    reader = CamRest676Reader()
    embeddings = read_embeddings(reader)
    model = SeqModel(vocab_size=cfg.vocab_size)
    prev_bspan_eos, bspan_eos, response_eos = "EOS_Z1", "EOS_Z2", "EOS_M"
    epochs = 10
    for epoch in range(epochs):
        data_iterator = reader.mini_batch_iterator('train')
        for iter_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            previous_bspan, previous_response = None, None
            for turn_num, turn_batch in enumerate(dial_batch):
                _, _, user, response, bspan_received, u_len, m_len, degree, _ = turn_batch.values()
                batch_size = len(user)
                if previous_bspan is None:
                    previous_bspan = [[reader.vocab.encode(prev_bspan_eos)] for i in range(batch_size)]
                    previous_response = [[reader.vocab.encode(response_eos)] for i in range(batch_size)]
                target_bspan, target_response = tensorize(bspan_received), tensorize(response)

                bspan_decoder_input = produce_bspan_decoder_input(previous_bspan, previous_response, user)
                response_decoder_input = produce_response_decoder_input(previous_bspan, previous_response,
                                                                        user, bspan_received, degree)
                # TODO write cleaner evaluate and train functions

                # TODO actually save the models, keeping track of the best one

                # training the model
                model.train_step_bspan(bspan_decoder_input, target_bspan)
                model.train_step_response(response_decoder_input, target_response)

                previous_bspan = [x if x != reader.vocab.encode(bspan_eos) else reader.vocab.encode(prev_bspan_eos)
                                  for x in [y for y in bspan_received]]
                previous_response = response
                print(model.response_loss.result(), model.bspan_loss.result())
