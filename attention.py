import numpy as np

np.random.seed(42)

ATTENTION_PARAMS = {}
FFN_PARAMS = {}


def softmax(z):
    z = np.array(z)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_scores = np.exp(z)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return probs


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(0, 2, 1)

    d_k = K.shape[-1]
    scores = scores / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores)
    output = attention_weights @ V

    return output, attention_weights


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def add_and_norm(X, sublayer_output):
    X_res = X + sublayer_output
    return layer_norm(X_res)


def relu(x):
    return np.maximum(0, x)


def init_attention_params(d_model, prefix="self_attention"):
    if prefix not in ATTENTION_PARAMS:
        ATTENTION_PARAMS[prefix] = {
            "W_Q": np.random.rand(d_model, d_model) * 0.01,
            "W_K": np.random.rand(d_model, d_model) * 0.01,
            "W_V": np.random.rand(d_model, d_model) * 0.01,
        }
    return ATTENTION_PARAMS[prefix]


def init_ffn_params(d_model, d_ff=256, prefix="ffn"):
    if prefix not in FFN_PARAMS:
        FFN_PARAMS[prefix] = {
            "W1": np.random.rand(d_model, d_ff) * 0.01,
            "b1": np.zeros(d_ff),
            "W2": np.random.rand(d_ff, d_model) * 0.01,
            "b2": np.zeros(d_model),
        }
    return FFN_PARAMS[prefix]


def feed_forward(X, params=None, prefix="ffn"):
    d_model = X.shape[-1]

    if params is None:
        params = init_ffn_params(d_model, prefix=prefix)

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    hidden = X @ W1 + b1
    hidden = relu(hidden)
    output = hidden @ W2 + b2
    return output


def self_attention(X, mask=None, params=None, prefix="self_attention"):
    d_model = X.shape[-1]

    if params is None:
        params = init_attention_params(d_model, prefix=prefix)

    W_Q = params["W_Q"]
    W_K = params["W_K"]
    W_V = params["W_V"]

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

    return output


def run_attention_demo():
    vector_array = [2.0, 1.0, 0.1]

    print("Teste softmax:", softmax(vector_array))

    X = np.random.rand(1, 10, 16)
    output = self_attention(X)

    print("\nShape da saída:", output.shape)


if __name__ == "__main__":
    run_attention_demo()