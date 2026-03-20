import torch
import torch.nn as nn

torch.manual_seed(42)


def softmax(z):
    return torch.softmax(z, dim=-1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1)

    d_k = K.shape[-1]
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores)
    output = attention_weights @ V

    return output, attention_weights


def layer_norm(x, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def add_and_norm(X, sublayer_output):
    X_res = X + sublayer_output
    return layer_norm(X_res)


def relu(x):
    return torch.relu(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):
        hidden = X @ self.W1 + self.b1
        hidden = relu(hidden)
        output = hidden @ self.W2 + self.b2
        return output


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, X, mask=None):
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, attention_weights


def run_attention_demo():
    vector_array = torch.tensor([2.0, 1.0, 0.1])

    print("Teste softmax:", softmax(vector_array))

    X = torch.rand(1, 10, 16)
    attention = SelfAttention(d_model=16)
    output, _ = attention(X)

    print("\nShape da saída:", output.shape)


if __name__ == "__main__":
    run_attention_demo()