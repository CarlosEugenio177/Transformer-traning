import numpy as np
from attention import self_attention, add_and_norm, feed_forward, softmax
from decoder import look_ahead_mask, cross_attention


def encoder_block(X, layer_id=0):
    X_att = self_attention(X, prefix=f"encoder_self_attention_{layer_id}")
    X_norm1 = add_and_norm(X, X_att)

    X_ffn = feed_forward(X_norm1, prefix=f"encoder_ffn_{layer_id}")
    X_out = add_and_norm(X_norm1, X_ffn)

    return X_out

def encoder_stack(X, num_layers=6):
    Z = X.copy()

    for layer_id in range(num_layers):
        Z = encoder_block(Z, layer_id=layer_id)

    return Z


def decoder_block(Y, Z, layer_id=0):
    mask = look_ahead_mask(Y.shape[1])

    Y_att = self_attention(Y, mask=mask, prefix=f"decoder_self_attention_{layer_id}")
    Y_norm1 = add_and_norm(Y, Y_att)

    Y_cross = cross_attention(Z, Y_norm1, prefix=f"decoder_cross_attention_{layer_id}")
    Y_norm2 = add_and_norm(Y_norm1, Y_cross)

    Y_ffn = feed_forward(Y_norm2, prefix=f"decoder_ffn_{layer_id}")
    Y_out = add_and_norm(Y_norm2, Y_ffn)

    return Y_out


def decoder_stack(Y, Z, num_layers=6):
    Y_out = Y.copy()

    for layer_id in range(num_layers):
        Y_out = decoder_block(Y_out, Z, layer_id=layer_id)

    return Y_out


def output_projection(Y, vocab_size, params=None):
    d_model = Y.shape[-1]

    if params is None:
        if not hasattr(output_projection, "W"):
            output_projection.W = np.random.rand(d_model, vocab_size) * 0.01
        W = output_projection.W
    else:
        W = params["W"]

    logits = Y @ W
    probs = softmax(logits)

    return probs


def run_inference():
    vocab = ["<START>", "Thinking", "Machines", "No", "i", "am", "your", "father", ".", "<EOS>"]

    encoder_input = np.random.rand(1, 2, 64)
    Z = encoder_stack(encoder_input)

    sequence = ["<START>"]
    Y = np.random.rand(1, 1, 64)

    while True:
        Y_dec = decoder_stack(Y, Z, num_layers=2)
        probs = output_projection(Y_dec, len(vocab))

        token_index = np.argmax(probs[0, -1])
        next_token = vocab[token_index]

        sequence.append(next_token)

        if next_token == "<EOS>":
            break

        new_vec = np.random.rand(1, 1, 64)
        Y = np.concatenate([Y, new_vec], axis=1)

        if len(sequence) > 20:
            break

    print("Entrada simulada do encoder: Thinking Machines")
    print("Sequência gerada:")
    print(" ".join(sequence))


if __name__ == "__main__":
    run_inference()