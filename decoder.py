import numpy as np
from attention import softmax, scaled_dot_product_attention

def look_ahead_mask(seq_len):

    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    mask = mask[np.newaxis, :, :]

    return mask


def cross_attention(encoder_out, decoder_state):

    d_model = encoder_out.shape[-1]

    W_Q = np.random.rand(d_model, d_model)
    W_K = np.random.rand(d_model, d_model)
    W_V = np.random.rand(d_model, d_model)

    Q = decoder_state @ W_Q
    K = encoder_out @ W_K
    V = encoder_out @ W_V

    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(d_model)

    attention = softmax(scores)

    output = attention @ V

    return output


def masked_self_attention(Y):

    d_model = Y.shape[-1]

    W_Q = np.random.rand(d_model, d_model)
    W_K = np.random.rand(d_model, d_model)
    W_V = np.random.rand(d_model, d_model)

    Q = Y @ W_Q
    K = Y @ W_K
    V = Y @ W_V

    mask = look_ahead_mask(Y.shape[1])

    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

    return output, attention_weights


def generate_next_token(current_sequence, encoder_out):

    vocab_size = 10000

    probs = np.random.rand(vocab_size)
    probs = probs / np.sum(probs)

    return probs


def autoregressive_loop():

    encoder_out = np.random.rand(1, 10, 512)

    vocab = ['No', 'i', 'am', 'your', 'father', '.', '<EOS>']

    sequence = ['<START>']

    while True:

        probs = generate_next_token(sequence, encoder_out)

        token_index = np.argmax(probs)

        next_token = vocab[token_index % len(vocab)]

        sequence.append(next_token)

        if next_token == '<EOS>':
            break

    print('Frase gerada:')
    print(' '.join(sequence))


if __name__ == '__main__':

    print('Teste máscara causal:')
    print(look_ahead_mask(5)[0])

    print('\nTeste cross-attention:')
    encoder_output = np.random.rand(1, 10, 512)
    decoder_state = np.random.rand(1, 4, 512)

    cross_output = cross_attention(encoder_output, decoder_state)
    print('Cross-attention shape:', cross_output.shape)

    print('\nTeste masked self-attention:')
    Y = np.random.rand(1, 4, 512)
    masked_output, masked_weights = masked_self_attention(Y)
    print('Masked self-attention output shape:', masked_output.shape)
    print('Masked self-attention weights shape:', masked_weights.shape)

    print('\nTeste loop auto-regressivo:')
    autoregressive_loop()