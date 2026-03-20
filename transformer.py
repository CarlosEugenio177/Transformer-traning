import numpy as np
from attention import self_attention, add_and_norm, feed_forward, softmax
from decoder import look_ahead_mask, cross_attention

def encoder_block(X):

    X_att = self_attention(X)
    X_norm1 = add_and_norm(X, X_att)

    X_ffn = feed_forward(X_norm1)
    X_out = add_and_norm(X_norm1, X_ffn)

    return X_out

def encoder_stack(X, num_layers=6):

    Z = X.copy()

    for _ in range(num_layers):
        Z = encoder_block(Z)

    return Z

def decoder_block(Y, Z):

    mask = look_ahead_mask(Y.shape[1])

    Y_att = self_attention(Y, mask)

    Y_norm1 = add_and_norm(Y, Y_att)

    Y_cross = cross_attention(Z, Y_norm1)
    Y_norm2 = add_and_norm(Y_norm1, Y_cross)

    Y_ffn = feed_forward(Y_norm2)
    Y_out = add_and_norm(Y_norm2, Y_ffn)

    return Y_out

def output_projection(Y, vocab_size):

    d_model = Y.shape[-1]

    W = np.random.rand(d_model, vocab_size)

    logits = Y @ W

    probs = softmax(logits)

    return probs
def run_inference():

    vocab = ['<START>', 'Thinking', 'Machines', 'No', 'i', 'am', 'your', 'father', '.', '<EOS>']

    encoder_input = np.random.rand(1, 2, 64)

    Z = encoder_stack(encoder_input)

    sequence = ['<START>']

    Y = np.random.rand(1, 1, 64)

    while True:

        Y_dec = decoder_block(Y, Z)

        probs = output_projection(Y_dec, len(vocab))

        token_index = np.argmax(probs[0, -1])

        next_token = vocab[token_index]

        sequence.append(next_token)

        if next_token == '<EOS>':
            break

        new_vec = np.random.rand(1, 1, 64)

        Y = np.concatenate([Y, new_vec], axis=1)

    print('Entrada simulada do encoder: Thinking Machines')
    print('Sequência gerada:')
    print(' '.join(sequence))


if __name__ == '__main__':
    run_inference()